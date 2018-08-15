# Conditional Random Field

## 结构

```sh
.
├── bigdata
│   ├── dev.conll
│   ├── test.conll
│   └── train.conll
├── data
│   ├── dev.conll
│   └── train.conll
├── result
│   ├── acrf.txt
│   ├── ascrf.txt
│   ├── crf.txt
│   ├── oacrf.txt
│   ├── oascrf.txt
│   ├── ocrf.txt
│   ├── oscrf.txt
│   └── scrf.txt
├── config.py
├── crf.py
├── ocrf.py
├── README.md
└── run.py
```

## 用法

```sh
$ python run.py -h
usage: run.py [-h] [-b] [--anneal] [--optimize] [--regularize] [--shuffle]

Create Conditional Random Field(CRF) for POS Tagging.

optional arguments:
  -h, --help        show this help message and exit
  -b                use big data
  --anneal, -a      use simulated annealing
  --optimize, -o    use feature extracion optimization
  --regularize, -r  use L2 regularization
  --shuffle, -s     shuffle the data at each epoch
  --file FILE, -f FILE  set where to store the model
# e.g. 特征提取优化+模拟退火+打乱数据
$ python run.py -b --optimize --anneal --shuffle 
```

## 结果

| 特征提取优化 | 模拟退火 | 打乱数据 | 迭代次数 |  dev/P   |  test/P  |     mT(s)      |
| :----------: | :------: | :------: | :------: | :------: | :------: | :------------: |
|      ×       |    ×     |    ×     |  78/89   | 93.5068% | 93.2273% | 0:46:45.832759 |
|      ×       |    ×     |    √     |  22/33   | 93.8671% | 93.6993% | 0:52:48.013512 |
|      ×       |    √     |    ×     |  30/41   | 93.6686% | 93.3879% | 0:48:04.814345 |
|      ×       |    √     |    √     |  19/30   | 94.1489% | 93.9187% | 0:48:21.350622 |
|      √       |    ×     |    ×     |  41/52   | 93.6719% | 93.4578% | 0:09:32.178284 |
|      √       |    ×     |    √     |  20/31   | 93.9771% | 93.8206% | 0:09:32.436235 |
|      √       |    √     |    ×     |  29/40   | 93.8120% | 93.5534% | 0:09:48.246697 |
|      √       |    √     |    √     |  34/45   | 94.2657% | 94.0989% | 0:09:43.802662 |