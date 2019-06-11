Code trích xuất một phần dữ liệu để test
```python
from six.moves import cPickle as pkl
from argparse import Namespace

params = Namespace(
    stopwords='stuff/vietnamese-stopwords.txt',
    save_path='build/gated_cnn.pickle',
    data_path='build/data.pickle',
    items_path='data/items_soha.306101.pickle',
    dictionary_path='build/dictionary.pickle',
)

items = pkl.load(open(params.items_path, mode='rb'))
test_items = dict()
cnt = 0
for itemId, item in items.items():
    test_items[itemId] = item
    cnt += 1
    if cnt == 1000:
        break


pkl.dump(test_items, open("build/test_items.pickle", "wb"))

```

#TODO
- Kiểm tra lại file pretrain word embedding xem có đúng định dạng có dấu _ chưa
- Tối ưu pretrain w2v model
- Thay đổi cấu trúc mạng

- Số bản ghi trong 1 ngày : 1427903