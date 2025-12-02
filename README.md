# 基于SEED数据集的脑电情绪识别项目

## 数据集介绍

####  1. **被试与实验设计**

- 共 **15 名被试**（编号为 1 到 15）

- 每名被试参与 **3 次独立实验 session**（基本间隔一周以上，减少记忆效应）

- 因此总共有：**15 人 × 3 sessions = 45 个实验文件**

  数据集文件结构如下：

  ```bash
  SEED/
  ├── 1/                 # 被试1
  │   ├── 1_20131027.mat  # session 1
  │   ├── 1_20131030.mat  # session 2
  │   └── 1_20131107.mat  # session 3
  ├── 2/
  │   ├── 2_20140404.mat
  │   └── ...
  ...
  └── 15/
  │   └── ...
  └──label.mat
  ```

---

#### 2. 单个 `.mat` 文件内容

 一个被试的一次session（比如 `1_xxx.mat`）包含 15 个trial

```json
dict_keys([
    '__header__', '__version__', '__globals__',
    'djc_eeg1', 'djc_eeg2', ..., 'djc_eeg15'
])
```

`djc_eeg1` 到 `djc_eeg15`：对应 **15 个电影片段 trial**

- 前缀 `djc` 是被试代号（不同被试前缀不同，如 `djc_eeg1`、`jl_eeg1` 等）

- 每个变量是一个 NumPy 数组，形状如下：

  ```json
  (62, T)
  ```

  - `62`：EEG 通道数（扩展 10–20 系统）
  - `T`：时间点数量（采样率 200 Hz × 视频时长约 4 分钟 → T ≈ 48,000）

- 这 15 个 trial 的情绪标签是**固定顺序**的（官方设定）

  ```json
  [1, 0, -1, -1, 0, 1, -1, 0, 1, 1, 0, -1, 0, 1, -1]
  ```

  - `1` = positive（积极）
  - `0` = neutral（中性）
  - `-1` = negative（消极）

> 📌 注意：**所有被试、所有 session 都观看完全相同的15个电影片段，使用相同的标签顺序**。

------

#### 3.官方数据预处理

根据 SEED 官网和原始论文，数据已经过以下处理：

- 采样率：**200 Hz**
- 滤波：**带通 0–75 Hz**

----

## 实验设计

根据我们的调研，情绪识别的核心目标是“泛化到新用户”，因此必须保证：

- 模型对没见过的人仍然有效
- 避免数据泄漏

所以我们以**跨被试**实验为主，训练数据采用`14个被试 `x `3session`，测试数据采用`1个被试` x `3session`。

同时使用 **留一法** 进行交叉验证。

-------

## 数据预处理

关于 **SEED 数据集的预处理**，目标是：**在保留情绪相关神经活动的同时，最大限度抑制噪声和个体差异**。

对于一个被试的一次session实验，先通过键名索引到最小的数据单元（即一次实验的电位数据），形状为 `(62, T)` ，记作 `X_raw` 。`T` 表示采样时间点的总长度。下面是对 `X_raw` 的数据预处理流程：（baseline）

1. 信号截取

   我们截取 `30-120 s` 时间段的信号

   目的：排除实验开始的适应期和结束时的疲劳期，提取到情绪体验比较稳定的EEG信号段。

2. 重参考-CAR(共同平均参考)

   （可以了解背景：为什么 EEG 需要“参考”？）

   数据发布前已经经过了一次参考：**统一转换为双乳突平均参考**

   ```py
   X_ref = X_raw - np.mean(X_raw, axis=0, keepdims=True)
   ```

   目的：我的理解是为了让不同通道间的真实相对差异**更清晰、更可靠**，做参考可能带来的好处就是[ **给所有数据去掉某个整体的偏移量** | **使得数据分布的空间不对称性更加清晰的体现** ]

3. 带通滤波

   保留情绪相关的生理频段（4-45Hz）

   目的：去除低频漂移、工频干扰和高频噪声

4. 频带分解

   ```json
   bands = {'delta':(1,4), 'theta':(4,8), 'alpha':(8,14), 'beta':(14,31), 'gamma':(31,50)}
   ```

   不同情绪在不同频带有特异性表现

   目的：分离不同生理意义的脑电节律

5. 微分熵计算

   **微分熵（DE）** 是衡量一个连续信号（如 EEG）**波动剧烈程度/信息量**的一个指标

   - **DE 越高** → 信号越“混乱”、能量越强（比如 beta 波活跃）

   - **DE 越低** → 信号越“平稳”（比如 alpha 波节律性强）

     （PS：在计算DE时，将原始通道的方差改成通道的功率power）

经过一系列的处理最终将原始的 `X_raw` 从 `(62, T)` -> `(310,)` 

------------

## 实验结果

`Mean_Accuracy` = 15个被试平均分类准确率

1. `DE` 微分熵特征提取 + `SVM` 分类模型

   作为 baseline 模型，进行了很多次的探索；综合前期多组实验的结果，确立基准模型，使用**网格搜索**超参数，并加入**归一化**操作，作为后续特征工程改进的起点。`Mean_Accuracy ` = 0.567

![subject_accuracy_lines(SVM_baseline)](https://gitee.com/jak-ma/graph-s/raw/master/imgs/20251202201926084.png)

​	无归一化时，`Mean_Accuracy`  = 0.532

![subject_accuracy_lines(SVM_baseline_non_sacler)](https://gitee.com/jak-ma/graph-s/raw/master/imgs/20251202202729648.png)

---

2. `DE` 微分熵特征提取+`mlp` 分类模型

   采用一个简单的 MLP 模块，构建这个的 baseline 时，主要是调参，具体就是`epochs `和 `学习率` 的把握，最终选择 `lr`  =  5e-4 |`epochs` = 30；因为没有设置验证集，所以采用的的方法是取最后5个 epoch的平均测试准确率作为衡量指标。

   `Mean_Accuracy`  = 0.569

![subject_accuracy_lines(MLP_with_scaler)](https://gitee.com/jak-ma/graph-s/raw/master/imgs/20251202211237567.png)

​	无归一化时，`Mean_Accuracy`  = 0.471

![subject_accuracy_lines(MLP_non_scaler)](https://gitee.com/jak-ma/graph-s/raw/master/imgs/20251202205931934.png)

由上述4组实验表明，归一化在模型训练过程中能够起到一定的正向作用，特别是对于MLP 来说。

## 算法改进

因为模型层面的改进已经在建立baseline时基本完成，接下来我们主要是在数据特征处理方面做了一些尝试。

1. PCA降维

   **"`DE` 微分熵特征提取 + `PCA` 降维+`SVM` 分类模型"**

   Variance(方差)：数据在某个方向上的 `spread` 程度，PCA 视其为`信息量`

   解释方差比例 = 前 k 个主成分的方差之和 / 原始总方差之和。Explained Var = 0.999：前 100 个主成分保留了原始数据 99.9% 的变化信息

   **保证降维后不丢失关键判别特征，同时去噪、提速、防过拟合**

   首次尝试 PCA 时设置 K=80，`Mean_Accuracy`  = 0.563↓

   ![subject_accuracy_lines(SVM_PCA80)](https://gitee.com/jak-ma/graph-s/raw/master/imgs/20251202210717436.png)

   显然说明现有数据预处理方法得到的特征存在大量冗余；接下来的话，尝试的方向就是找出最优K值。

   有两种方法，一种是直接采用一个数组存放一些候选的K值，然后遍历它，做多次实验，通过模型表现过来验证；而是找**累计**解释方差曲线（类似下图）的拐点。

   <img src="https://gitee.com/jak-ma/graph-s/raw/master/imgs/20251202162024400.png" alt="Explain_Var" style="zoom:50%;" />

   但是因为我采用的留一法，我感觉对于每一个被试去采用不管是法1还是法2，消耗的资源成本大于收益，所以可以使用二分法进行试错。

   K = 100时，`Mean_Accuracy`  = 0.551↓，变得更差了

   ![subject_accuracy_lines(SVM_PCA100)](https://gitee.com/jak-ma/graph-s/raw/master/imgs/20251202211854716.png)

   K = 60，`Mean_Accuracy`  = 0.548↓

   ![subject_accuracy_lines(SVM_PCA60)](https://gitee.com/jak-ma/graph-s/raw/master/imgs/20251202215211745.png)

   显然，PCA降维对于SVM分类模型并没有提升。

   ---

   **"`DE` 微分熵特征提取 + `PCA` 降维+`MLP` 分类模型"**

   K = 80，`Mean_Accuracy`  = 0.583&uarr;

   ![subject_accuracy_lines(MLP_PCA80)](https://gitee.com/jak-ma/graph-s/raw/master/imgs/20251202213318478.png)

   K = 100，`Mean_Accuracy`  = 0.552↓

   ![subject_accuracy_lines(MLP_PCA100)](https://gitee.com/jak-ma/graph-s/raw/master/imgs/20251202213733259.png)

   K = 60，`Mean_Accuracy`  = 0.569↓

   ![subject_accuracy_lines(MLP_PCA60)](https://gitee.com/jak-ma/graph-s/raw/master/imgs/20251202214300691.png)

   最终经过尝试，PCA降维 是对 MLP分类模型 有效的，选择的最佳K=80;

   分析原因，为啥 PCA降维 对于 SVM分类模型 没有提升呢？

   ---

2. ANOVA (方差分析) 特征选择

   ANOVA（方差分析）用来检验：不同类别（情绪）的数据，在该特征上是否有显著差异。用 `F值` 来衡量：

   `F值`  =  组间方差 / 组内方差，`F值`越大，说明特征越有区分力

   n_features：表示选出前 n_features 个较大 `F值`  

   经过小部分实验，我认为 `方差分析` 降维方法在这两个分类模型上都没有较大可用性

   | experiment | MLP_ANOVA(80) | MLP_ANOVA(160) | SVM_ANOVA(80) |
   | ---------- | ------------- | -------------- | ------------- |
   | Accuracy   | 0.533         | 0.549          | 0.529         |

---

由上我们能够分析出来使用微分熵特征提取手段，得到的特征其实是存在大量的冗余，所以这里还可以去尝试其他特征提取方式—如	**小波变换**，在频域上进行特征提取等等，还有很多可以尝试的方向。



