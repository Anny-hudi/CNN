# (Re-)Imag(in)ing Price Trends — 论文完全复现说明

## 目标
按 Jiang, Kelly & Xiu (2023) 在 *The Journal of Finance* 的实验，实现论文所有关键实验并得到可比结果：
- I5/R5, I20/R20, I60/R60 等 CNN 模型的样本外（2001–2019）组合表现；
- 对比 MOM / STR / WSTR / TREND 四种基准；
- 等权 (equal-weight) 与 价值加权 (value-weight) 分位组合、H-L（decile10 - decile1）；
- 报告平均持有期回报、年化 Sharpe、月度换手率、累计净值曲线；
- 迁移学习实验（美国训练模型应用于国际市场）为可选扩展。

---

## 数据（严格按照论文）
- **数据源**：CRSP 日频 OHLCV（NYSE、AMEX、NASDAQ 全部公司）
- **样本区间**：
  - 可用 OHLC（含 open/high/low）最早从 1992-06 开始，论文训练/测试分配如下：
  - 训练/验证期：**1993-01-01 — 2000-12-31**（对整个训练区间随机 70%/30% 划分为训练/验证）
  - 测试（样本外）：**2001-01-01 — 2019-12-31**
- **字段**：date, permno/stock_id, open, high, low, close, volume
- **价格预处理（与论文一致）**：
  - 用 CRSP-adjusted returns 构造价格路径：在每个图像窗口里令首日收盘价 = 1，然后用后续收益递推 `p_{t+1} = (1 + RET_{t+1}) * p_t`（论文描述）
  - 当窗口内出现缺失（例如 IPO / delist）处理：如论文，排除在样本窗口开头或结尾存在 IPO/delist 的图像；中间缺失的日对应列留空（或画黑）
- **图像生成细则（严格）**：
  - 窗口长度三种：**5、20、60** 日
  - 每日占 **3 像素宽**（中心竖条 + 左开盘标记 + 右收盘标记）→ 图像宽度 = `3 * window`
  - 图像高度：固定（论文未给确切像素高度），推荐 `img_h = 64`（可调）；纵轴缩放：使该窗口内 OHLC 的 max/min 对齐到 图像 top/bottom（保证跨股票可比）
  - 颜色/像素：黑底（0），可见对象白色（255）。只用灰度单通道。
  - 成交量条：放在图像底部的 1/5 区域（height * 1/5），按该窗口最大成交量归一化绘制
  - 移动平均线：窗口长度对应的 MA（例如 20 日图像画 20 日 MA），在中列处每日日像素画点并连线
  - 缺失数据处理：若高/低缺失则该日列黑（论文做法）；若高低有但 open/close 缺失则只画竖线

---

## 标签与预测任务
- **任务**：二分类（未来持有期累计回报是否为正）
- **标签**：`y = 1` 若持有期累计收益 > 0；否则 `y = 0`
- **监督/持有期对应关系**（论文的九个模型）：
  - I5/R5：使用过去 5 日图像预测接下来 5 日是否为正回报（周策略）
  - I20/R20：过去 20 日图像预测接下来 20 日（约月）
  - I60/R60：过去 60 日图像预测接下来 60 日（约季度）
  - 另外论文也训练过不同 supervise/hold combinations（例如用短期监督去预测长周期），可复现实验里包含这些扩展

---

## CNN 架构与训练超参数（严格）
- **输入**：灰度单通道图像，尺寸 `(1, img_h, 3*window)`
- **架构总览（按论文图示）**：
  - I5 模型：**2** 个卷积“block”
  - I20 模型：**3** 个卷积“block”
  - I60 模型：**4** 个卷积“block”
- **一个卷积 block**：
  - `Conv2d(in_ch, out_ch, kernel_size=3, padding=1)`  
  - `BatchNorm2d(out_ch)`  
  - `LeakyReLU()`  
  - `MaxPool2d(kernel_size=2)`  
- **通道深度**（paper 指出示例通道）：`[64, 128, 256, 512]`（依 block 级数选择前 N 个）
- **全连接层**：
  - Flatten → Linear → Dropout(0.5) → ReLU → Linear(2) → Softmax（输出上/下概率）
- **训练细节**：
  - 损失：`CrossEntropyLoss`（论文用交叉熵）
  - 优化器：Adam（Kingma & Ba）
  - 初始学习率：`1e-5`
  - batch size：`128`
  - 权重初始化：Xavier (Glorot) 初始化
  - batch normalization：在每个卷积块中使用
  - dropout：**50%**，只在全连接层
  - early stopping：在验证集上若验证 loss 连续 **2 个 epoch** 无改善则停止训练
  - 随机性：对每个模型配置**独立训练 5 次**（不同随机初始化 &数据 shuffle），然后对预测概率求**平均**作为最终预测（论文方法）
  - 训练/验证划分：**在 1993–2000 时段内随机 70%/30%（样本随机抽取，不按时间切分）**（论文如此设计以平衡 label 分布）
  - 测试期：训练好模型后固定权重，在 2001–2019 上做样本外预报（不进一步再训练）

---

## 基准信号（与论文一致）
- **MOM**：2–12 月动量（通常去掉最近 1 个月）。论文用作比较基准。
- **STR**：1 个月短反转（short-term reversal）
- **WSTR**：1 周短反转（weekly short-term reversal）
- **TREND**：Han, Zhou & Zhu (2016) 的组合短/中/长期趋势信号
> 注：要严格复现这些基准，需要用论文或引用文献给出的确切实现细节（例如动量的 lookback/window/skip month 是否精确如 in paper）。复现时请严格按论文中他们用于基准的实现参数来实现（若论文 Internet Appendix 有具体代码/伪代码，应以其为准）。

---

## 投资组合构建（严格）
- **每个预测期**（例如周/月/季），按样本外 CNN 预测概率对所有股票做横截面排序，分成 10 个十分位（deciles）
- **等权组合**：在每个再平衡期内，对 decile 1..10 分别等权持有（等权 H-L 为 decile10 long, decile1 short）
- **价值加权**：以该期的市值为权重构建 decile 投资组合，H-L 同理（long decile10, short decile1）
- **组合的持有期**与模型的预测 horizon 一致（I5/R5 持有 5 天，I20/R20 持有 20 天，I60/R60 持有 60 天）
- **每次再平衡**按样本外预测重新排名并重构组合（paper 的 weekly/monthly/quarterly策略均如此）

---

## 评估指标与计算公式（严格按论文）
- **持有期回报**：等权 / 价值权组合在该持有期的实际对数回报或算术回报（与论文一致）
- **H-L 回报**：`R_HL = R_decile10 - R_decile1`
- **年化化 Sharpe Ratio**（paper 用年化 Sharpe）：
  - 若使用频率为 `f`（每年周期数：weekly f=52, monthly f=12, quarterly f=4）
  - `Sharpe = mean(period_returns) / std(period_returns) * sqrt(f)`
  - 论文报告的是年化 Sharpe（未减去无风险利率），直接用上式（若要精确论文值，确保返回序列与论文用的 return 类型一致：比如等权/市值权算术 return）
- **换手率 (Turnover)**（论文给公式）：
  - 按 Gu, Kelly & Xiu (2020) 的定义（也在论文中给出）：
  - `Turnover = (1/M) * (1/T) * sum_{t=1..T} sum_i | w_{i,t+1} - w_{i,t} * (1 + r_{i,t+1}) / (1 + sum_j w_{j,t} r_{j,t+1}) |`
  - 其中 M = 持有期月数（例如持有 1 周 ~ 1/4 月），T = 期数，w_{i,t} 为 t 时刻的净权重
- **交易成本调整（可选）**：
  - 论文在某些扩展中调整标准交易成本。复现时可将 `turnover * cost_per_trade` 从组合回报中扣除（cost_per_trade 依据你的选择或论文所用值）

---

## 可视化与解释
- 绘制等权 H-L 的累计对数收益曲线（paper 图表形式）
- 绘制不同 decile 的平均回报与时间序列波动（paper Figure 6）
- 可视化 CNN 特征/激活映射（Grad-CAM / activation maps）以解释模型学习的图形模式（论文用两种方法分析 CNN 解释）

---

## 复现注意事项 & 潜在差异来源
- 图像像素高度 `img_h`、像素化细节（是否画 1 像素宽的 open/close tick vs thicker）、以及 volume bar 的确切绘制方式会影响训练输入——这会导致数值上差异但通常不改变结论方向
- 训练/验证随机划分（paper 在训练期内随机抽样）会引入一定随机性，论文通过 5 次独立训练取平均来稳定
- 论文在 Internet Appendix 可能给有更多实现细节与稳健性检验，务必对照 Internet Appendix（若可获得）
