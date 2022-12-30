import categorical_preprocessing as cp

sage = cp.Categorical_Preprocessing()
sage.read_data('api_data.csv')
sage.drop_columns(['timestamp', 'meta.host', 'meta.job',
                   'meta.node', 'meta.plugin', 'meta.task'])
sage.numeric('value')
maps = {'W02C': 0, 'W07A': 1, 'W07B': 2, 'W026': 3, 'W079': 4}
sage.replace_mapping(['nodes'], maps)
sage.unique_values()
sage.unique_col_vals('name')
sage.dummies('name')
sage.drop_columns(['name', 'env.count.aeroplane', 'env.count.airplane', 'env.count.apple',
                   'env.count.backpack', 'env.count.banana', 'env.count.bench',
                   'env.count.bird', 'env.count.boat', 'env.count.book', 'env.count.bottle',
                   'env.count.carrot', 'env.count.cat', 'env.count.cell_phone', 'env.count.chair',
                   'env.count.clock', 'env.count.cow', 'env.count.dog', 'env.count.elephant',
                   'env.count.fire_hydrant', 'env.count.giraffe', 'env.count.handbag',
                   'env.count.horse', 'env.count.keyboard', 'env.count.kite', 'env.count.knife',
                   'env.count.microwave', 'env.count.mouse', 'env.count.parking_meter', 'env.count.pottedplant',
                   'env.count.remote', 'env.count.scissors', 'env.count.sink', 'env.count.skateboard', 'env.count.skis',
                   'env.count.snowboard', 'env.count.stop_sign', 'env.count.suitcase',
                   'env.count.surfboard', 'env.count.tennis_racket', 'env.count.tie', 'env.count.toilet',
                   'env.count.traffic_light', 'env.count.train', 'env.count.truck', 'env.count.tvmonitor',
                   'env.count.umbrella', 'env.count.zebra', 'env.count.refrigerator', 'env.count.teddy_bear',
                   'env.count.baseball_glove', 'env.count.wine_glass', 'env.count.bear','env.count.cup',
                   'env.count.laptop','traffic.state.flow', 'traffic.state.log', 'traffic.state.occupancy'])
onehot_columns = ['env.count.bicycle', 'env.count.bus', 'env.count.car', 'env.count.motorbike', 'env.count.motorcycle',
                  'env.count.person', 'traffic.state.averaged_speed']
onehot_prefixes = ['b', 't', 'c', 'm', 'v', 'p', 'a']
sage.onehot_encode(onehot_columns, onehot_prefixes)
sage.replace_mapping(onehot_columns, maps)
sage.convert_int()
sage.data_type()
data = sage.copy()

data.to_csv('data.csv', index=False)

