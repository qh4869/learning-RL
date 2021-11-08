### 环境 ###

- 同一行不断刷新
  - print的end参数最后默认是\n，改为空字符串不换行
  - \r表示光标回到行首



### 训练 ###

- episode：一次完整的交互
- step: 每个action对应的一步



### 数据结构 ###

- 表格: panda模块
  - panda.DataFrame(numpy/list, index, column)
  - DataFrame的每一行是panda.Series
  - 不管是行还是表格，都可以用data.loc[idx]来索引
- numpy:
  - numpy.zeros
  - numpy.random.XXX
  - numpy是可以类似于matlab中a[a>0]这样索引的，它做了重载，传入的属性是bool的list or idx的list它有不同的处理。（list类就没有这样的重载）

