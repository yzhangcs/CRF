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
```

## 结果

| 特征提取优化 | 模拟退火 | 打乱数据 | 迭代次数 |  dev/P   |  test/P  |     mT(s)      |
| :----------: | :------: | :------: | :------: | :------: | :------: | :------------: |
|      ×       |    ×     |    ×     |  77/88   | 93.5068% | 93.2273% | 0:58:55.903059 |
|      ×       |    ×     |    √     |  26/37   | 93.9138% | 93.7777% | 0:58:10.217997 |
|      ×       |    √     |    ×     |  29/40   | 93.6686% | 93.3879% | 1:00:06.907741 |
|      ×       |    √     |    √     |  28/39   | 94.2157% | 93.9224% | 1:03:03.541540 |
|      √       |    ×     |    ×     |  37/48   | 93.6769% | 93.4455% | 0:10:15.932644 |
|      √       |    ×     |    √     |  33/44   | 94.0706% | 93.9077% | 0:10:40.005814 |
|      √       |    √     |    ×     |  26/37   | 93.8070% | 93.5105% | 0:10:38.460952 |
|      √       |    √     |    √     |  23/34   | 94.2640% | 94.0241% | 0:10:55.674318 |