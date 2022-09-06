# Data Format
The data should be such format:
```
/
├── data/                   --> (The folder containing used datasets)   
|   ├── ml-1m/              --> (The ml-1m dataset for recommendation.)
|   |   ├── train_data.data   --> (The train data.)
|   |   ├── valid_data.data   --> (The valid data.)
|   |   ├── test_data.data    --> (The test data.)
|   ├── taobao/             --> (The taobao dataset for recommendation.)
```

## .data format

Since the datasets have some attrs, for one instance, let the format be like this:

user_id, item_id, user_attrs_i...(user tower), item_attrs_i...(item tower), user_attrs_i..., item_attrs_i...(user-item tower), label