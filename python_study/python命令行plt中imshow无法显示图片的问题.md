# python命令行plt中imshow无法显示图片的问题

以下代码在命令行中无效：

```python
import matplotlib.pyplot as plt
plt.imshow(img)
```

需引入pylab包

```python
import matplotlib.pyplot as plt
import pylab

plt.imshow(img)
pylab.show()
```

