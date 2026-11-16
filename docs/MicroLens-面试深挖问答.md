# MicroLens 多模态序列推荐｜面试深挖问答

> 文中说“模拟测试”时，用的是最终报告里的模拟数字。验证集的召回和 zero-shot 结果沿用项目归档。

## 1. 讲解你这个项目，包括设计，功能，具体的实现，公式，技术栈等等

**中文口述版，约 2 分钟**

> 这个项目做的是短视频推荐。给我一个用户最近看的最多五条视频，我想预测他下一条会看什么。MicroLens 有 19,220 条视频，把每条都交给 Qwen 看一遍太慢，所以我们先用 CLIP 找 50 个候选，再让 Qwen 排一下顺序。
>
> 召回这一步，我会提前用 CLIP 把每条视频的标题和图片转成向量，然后各占一半合起来。给候选打分时，我先看它和哪条历史视频最像，这个相似度占 75%。再看它和用户最后看的一条有多像，这部分占 25%。比如用户既看跑步也看烘焙，两种兴趣都能捞到候选，最近看的内容还能多一点权重。这个改动把验证集的 Recall@50 从 **6.48% 拉到了 7.14%**。
>
> 重排时，我们把历史标题和一条候选的标题、图片交给 Qwen2.5-VL-3B，问它用户接下来会不会看这条。模型其实不用开口生成，直接拿第一步的 Yes 分减 No 分，差越大排得越靠前。我主要负责训练和评估这部分。先用 LoRA 做 SFT，教模型什么时候该答 Yes 或 No，再用 DPO 调整它对这两个答案的偏好。每个用户用真实下一条视频当正例，再从 CLIP 召回里挑四条长得像的难负例。
>
> 报告的模拟测试里，同样是 50 个候选，CLIP 原排序的前十命中率是 **2.86%**，直接用 Qwen 重排降到了 **2.32%**，用难负例做 DPO 后是 **3.58%**。把候选图拿掉后，DPO 组回到 **3.36%**。工程上，召回用 NumPy，训练和推理用 PyTorch、Transformers 和 PEFT。

**English version, about 90 seconds**

> We built a short-video recommender. Given up to five videos from a user’s recent history, we try to predict what they’ll watch next. MicroLens has about 19,000 videos, so we use CLIP to get 50 candidates first and then let Qwen rerank them.
>
> We cache a CLIP embedding for each video by combining its title and image. When we score a candidate, 75% of the score comes from its best match anywhere in the user’s history, and 25% comes from its match to the most recent video. That took validation Recall@50 from 6.48% to 7.14%.
>
> My part was training and evaluation. Qwen sees the history titles and one candidate’s title and image. We rank candidates by the difference between the Yes and No logits. We use the actual next video as the positive and sample four hard negatives from CLIP’s shortlist. We start with LoRA supervised fine-tuning and then use DPO to adjust the Yes/No preference.
>
> In the report’s simulated Top-50 test, the hit rate at 10 went from 2.86% with CLIP to 2.32% with zero-shot Qwen. DPO with hard negatives brought it to 3.58%. Taking away the candidate image brought it down to 3.36%. We used NumPy for retrieval and PyTorch, Transformers, and PEFT for training and inference.

**白板公式**

~~~text
e_i = L2Normalize(0.5 · t_i + 0.5 · v_i)
s_CLIP(u,i) = 0.75 · max_j(e_hj · e_i) + 0.25 · (e_hm · e_i)
s_Qwen(u,i) = log P(Yes | H_u, i) − log P(No | H_u, i) = z_Yes − z_No
~~~

这里的 `t_i` 和 `v_i` 是已经归一化的标题、图像向量，`h_m` 是最近一次交互。

[召回实现](mm-retrieval-rerank-fork/src/mmrr/interest.py) · [Qwen 打分实现](mm-retrieval-rerank-fork/src/mmrr/models/qwen_scorer.py) · [模拟结果总表](mm-retrieval-rerank-fork/README.md#end-to-end-study)

## 项目范围、数据和个人贡献

### 2. 这是组队项目，哪些是你负责的？从哪段代码讲得最有把握？

> 我主要负责从训练数据到重排评估这段。具体来说，我把召回候选接进训练，做了随机负例和难负例两套数据，再接上 LoRA SFT、DPO、打分和评估。召回方案和最早的 zero-shot 对照是大家一起做的。
>
> 如果现场看代码，我会从 `prepare → train → score → evaluate` 开始。比如我可以直接展示，同一个用户怎么生成两组负例，又怎么让几个模型对同一份候选打分。这样最后看到差异时，才知道比较的是什么。

[训练评估入口](mm-retrieval-rerank-fork/scripts/pipeline.py) · [后半程运行手册](mm-retrieval-rerank-fork/docs/final_pipeline.md)

### 3. 你用的是什么数据？简历里的 products 是电商数据吗？

> 这里的 item 指的是短视频。我们用的是 HaNoRec 整理过的 MicroLens，目录里有 19,220 个物品，每个有 ID、标题和图。训练、验证、测试分别有 **4 万、5 千、5 千** 条用户序列，用户互不重叠。
>
> 我们沿用数据原有的顺序，把最后一条当答案，往前最多取五条当历史。报告里提到的 8,192 是模拟训练方案里每组抽取的序列数，4 万是整个训练集的规模。

[数据加载器](mm-retrieval-rerank-fork/src/mmrr/data/loaders.py) · [期中报告](<Project Midterm Report.pdf>)

### 4. 怎么避免数据泄漏？为什么召回脚本能读取含有 target 的文件？

> 同一行里有历史，也有正确答案，但代码先用历史和目录里的标题、图片把候选排出来。排完之后，评估才拿 target 去查它有没有进榜。就算真实目标进了候选，Qwen 看到的也只是它的标题和图片，跟其他候选一样。
>
> 训练样本和流行度都从训练集来。我们用验证集挑配置，再把固定下来的配置拿去做测试。人工把答案加进候选是另外一个诊断，专门看目标已经在场时模型会怎么排。

[数据和提示词接口](mm-retrieval-rerank-fork/docs/interfaces.md) · [评估实现](mm-retrieval-rerank-fork/src/mmrr/eval/experiment.py)

## Stage 1｜向量召回

### 5. 为什么用两阶段？为什么不全库用 Qwen？

> 因为两种模型的算力成本差很多。CLIP 的向量可以提前算好，请求来了只做点积。Qwen 每看一条候选，都得把这个用户的历史和候选图文读一遍。先用 CLIP 取 Top-50，Qwen 每个用户就只打 50 次分，全库打分得做 19,220 次，次数差了大概 **384 倍**。这个做法的代价是，第一步漏掉的目标，第二步就排不到了。

[项目提案](<Project Proposal.pdf>) · [期中报告](<Project Midterm Report.pdf>)

### 6. CLIP 本身是怎么训练的？项目怎样利用它？

> 可以把 CLIP 想成两个编码器，一个读图，一个读文字。训练的时候，它让配对的图和文字向量更接近，也让一张图和同一批里没配对的文字离得更远。
>
> 我们直接用现成的 CLIP ViT-B/32。标题和图片各出一个 512 维向量，先把它们各自归一化，五五开加在一起，再归一化一次。这样召回时，两条视频做一次点积，就能得到它们的余弦相似度。

[CLIP 原论文](https://proceedings.mlr.press/v139/radford21a.html) · [编码器实现](mm-retrieval-rerank-fork/src/mmrr/models/clip_encoder.py)

### 7. 为什么 75% max + 25% recent？和 α、γ 有什么关系？

> 比如用户最近既看跑步也看烘焙。把历史压成一个平均向量，两种兴趣容易互相稀释。所以我让每条候选自己去找最像的那条历史，这个分占 75%。剩下 25% 看它和最近一次有多像，给近期兴趣加点权重。
>
> α 管一条视频里标题和图片各占多少，我们用 0.5。γ 是老方案里给历史做时间衰减用的。现在直接算最高相似度和最近一次相似度，所以公式里只用前面的 75/25。验证集上 75/25 的 Recall@50 是 **7.14%**，50/50 的前十命中率则略高，**3.10% 对 3.02%**。我们按召回来挑，因为后面还有重排。

[历史兴趣召回对照](mm-retrieval-rerank-fork/reports/retrieval/history-interest-report.md)

### 8. 能讲一遍 exact Top-K 的代码和复杂度吗？

> 先按物品 ID 找到用户的历史向量，和整个目录矩阵做一次乘法。比如历史有五条、库里有 19,220 条视频，就得到五行相似度。对每一列取最高分和最后一行的分，按 75/25 加起来。把看过的那几条排除，剩下的按分数降序取前 K 个。同分就按物品 ID 由小到大排。
>
> 写成复杂度，历史长度是 `m`，目录大小是 `N`，向量维数是 `d`。点积是 `O(mNd)`，代码用了完整排序，所以排序是 `O(N log N)`。

[矩阵检索实现](mm-retrieval-rerank-fork/src/mmrr/interest.py)

### 9. 为什么选 K=50？K=100 或采样更加大的 K 会如何？

> 50 是我们的主实验预算。验证集上，从 50 扩到 100，召回从 **7.14% 到 9.98%**，多找回 142 个目标，但每个用户的 Qwen 打分次数翻倍。
>
> 模拟测试里，扩大到 100 后，难负例 DPO 的 HR@10 从 **3.58% 到 3.72%**，重排 P95 从 **5.58 秒到 10.96 秒**。所以候选可以继续加，召回也会涨，但加进来的视频还得和原来的候选竞争前十，最终进榜的收益可能小很多。

[召回验证对照](mm-retrieval-rerank-fork/reports/retrieval/history-interest-report.md) · [模拟候选预算对照](mm-retrieval-rerank-fork/README.md#image-ablation-negative-difficulty-and-candidate-budgets)

### 10. 图像和文本的各自贡献是什么？多路召回试过吗？

> 单看验证集的 Recall@50，只用标题是 **3.74%**，只用图是 **4.46%**，把图和文字合起来做折扣历史召回是 **6.48%**。这份数据上，图像单独就有用，和文字合起来还能找回更多。
>
> 我们也试了三路各取一些，图文一起取 30，标题和图片各取 10，去重、补齐，再按各路名次融合。结果是 **6.40%**。它找回了原方案漏掉的 47 个目标，同时丢了原方案找得到的 51 个，最后少了 4 个。

[期中报告](<Project Midterm Report.pdf>) · [多路召回报告](mm-retrieval-rerank-fork/reports/retrieval/multiroute-report.md)

### 11. 你们用 FAISS 了吗？把 19,220 个物品放大到百万级会怎么处理？

> 主结果用 NumPy 扫全库。代码也接了 FAISS 的 `IndexFlatIP`。它先给每条历史各找一小批最相似的物品，把这些候选合起来，再按 75/25 重算。每条历史的内积查找是精确的，但各自截断之后，有可能漏掉综合起来分更高的物品。
>
> 到百万级，我会用全库精确结果当参照，试近似向量索引。主要看它能保住多少 Recall@K，以及 P95 和内存能降多少，再决定每条历史应该先取多少候选。

[FAISS 实现](mm-retrieval-rerank-fork/src/mmrr/interest.py)

## Stage 2｜Qwen 重排和微调

### 12. 给我一个实际 prompt。怎么把 Yes/No 转成可排序的分数？

> 默认用 `onerec@v1`。比如历史里放《入门跑鞋测评》和《十公里训练》，候选是《半马训练计划》，再附上候选的封面。最后一句是 “Will the user watch the candidate video next? Answer Yes or No.”
>
> Yes 和 No 都是单个 token，所以我直接拿第一步的 Yes logit 减 No logit。差越大，就越倾向于把这条往前排。这个差值也等于 `log P(Yes) − log P(No)`，因为两边的 softmax 分母一样，做差就消掉了。
>
> 工程上我们直接用 FP32 算 `hᵀ(W_Yes − W_No)`。这样可以少一些 BF16 舍入造成的同分，模型主干还是用 BF16。

[提示词模板](mm-retrieval-rerank-fork/src/mmrr/models/qwen_prompt.py) · [Qwen 打分实现](mm-retrieval-rerank-fork/src/mmrr/models/qwen_scorer.py)

### 13. 这是 pointwise、pairwise，还是 listwise？为什么这样做？

> 推理是 pointwise。每次给 Qwen 看一条候选，50 条分别拿到分数，最后统一排。这样一次只放一张候选图，显存和上下文长度比较好控制，也好把候选分批送进模型。代价是模型看不到候选之间的关系，用户历史也会重复输入。
>
> DPO 里确实有一对东西。那一对是同一条候选的 Yes 和 No 两个答案。到了推理，还是每条候选做一次前向就能拿到分。

[提示词和数据实现](mm-retrieval-rerank-fork/src/mmrr/data/prompt_dataset.py)

### 14. 为什么用难负例？怎么构造，如何保持公平？

> 随机抽四条视频，往往一眼就能看出和用户没关系。但 Qwen 真正要排的，是 CLIP 已经筛过的一批相似视频，所以我也让它在训练里见见这种难题。
>
> 两组用同一批用户、同一个正例，每人都取四个负例。随机组从全库抽，难负例组从召回前 50 抽。抽的时候排除这个用户在训练序列里已经交互过的全部视频。谁有一组凑不满四个，两组都跳过。这里的负例是“没观察到互动”，所以也可能有用户本来会喜欢的视频。
>
> 模拟测试里，两组 DPO 从同一个 SFT 起点出发，难负例组的 HR@10 是 **3.58%**，随机组是 **3.34%**。只从第 1–10 名抽是 **3.54%**，只从第 41–50 名抽是 **3.34%**。

[样本构造](mm-retrieval-rerank-fork/src/mmrr/prepare.py) · [模拟负例对照](mm-retrieval-rerank-fork/README.md#image-ablation-negative-difficulty-and-candidate-budgets)

### 15. 真实下一次交互如果没被召回，还能当训练正例吗？DPO 输入长什么样？

> 可以。训练时，我直接拿这个用户下一条实际看过的视频当正例。召回在这里负责找负例。比如 A 是下一条实际看的视频，那 A 的 prompt 里，chosen 是 Yes，rejected 是 No。如果 B 是一个没观察到交互的难负例，B 的 prompt 里就反过来，chosen 是 No，rejected 是 Yes。
>
> 所以每个用户有五条记录，一条正例、四条负例。每条记录都在同一条视频上比较 Yes 和 No。到了评估，候选是固定的，漏召回的目标就计零分。

[训练记录生成](mm-retrieval-rerank-fork/src/mmrr/prepare.py)

### 16. 解释 LoRA 公式。哪些参数训练，哪些参数冻结？

> 原来有个大矩阵 `W₀`。我们把它冻住，旁边加两个小矩阵 A 和 B。前向时，就相当于用 `W₀ + (α_LoRA / r) · BA`。因为中间只经过 r 个维度，训练的参数量就从 `d_out × d_in` 变成 `r(d_in + d_out)`。
>
> 项目里 r 是 8，α 是 32。LoRA 加在 Qwen 语言部分的注意力和 MLP 投影上，只更新 adapter。底座和视觉编码器都冻结。省下来的是可训练参数和对应的梯度、优化器状态，模型本身还是要放进显存做前向。

[LoRA 原论文](https://arxiv.org/abs/2106.09685) · [训练实现](mm-retrieval-rerank-fork/src/mmrr/train.py)

### 17. SFT 和 DPO 的 loss 分别是什么？reference 怎么得到？

> SFT 就是告诉模型这道题答案是 Yes 还是 No，然后用交叉熵 `−log P(正确答案)`，把正确答案的概率往上推。
>
> DPO 看的是，我现在对正确答案的偏好，比 SFT 时强了多少。举个例子，正确答案是 Yes，就先算当前模型的 Yes 分减 No 分，再减掉 SFT 模型的那个差值。如果正确答案是 No，就把方向反过来。
>
> reference 就是同一个种子的随机负例 SFT 模型。实现时先把 SFT adapter 合进底座，再挂一层新的 DPO adapter。把新 adapter 关掉就是 reference，打开就是正在训练的模型。两组 DPO 随机和难负例都用这个起点。

**白板公式**

~~~text
m = z_Yes − z_No
L_DPO = −log σ(β · y · (m_policy − m_reference))
y = +1 if chosen is Yes, otherwise −1
β = 0.1
~~~

[DPO 原论文](https://arxiv.org/html/2305.18290#S4) · [损失实现](mm-retrieval-rerank-fork/src/mmrr/models/preference.py)

### 18. 用同一批二元标签做 SFT 再做 DPO，DPO 一定更好吗？

> 要看同一批候选上的结果。SFT 让模型把正确答案的概率提上去。DPO 是从 SFT 出发，继续调整 Yes 和 No 拉开多少。两步在用同一种反馈，所以第二步有没有收益得靠对照。
>
> 模拟测试里，随机负例的 SFT 是 **3.10%**，从它接着做随机负例 DPO 到 **3.34%**，改用难负例做 DPO 到 **3.58%**。单做难负例 SFT 是 **3.24%**。这些数在同一个 Top-50 测试口径下。要单独看 DPO 阶段换负例的影响，就比 **3.34% 和 3.58%**，因为它们从同一个 SFT 出发。

[实验配置](mm-retrieval-rerank-fork/configs/final_experiments.json) · [模拟结果总表](mm-retrieval-rerank-fork/README.md#end-to-end-study)

### 19. HaNoRec 在这里做了什么？和原论文的方法一样吗？

> 我们用了 HaNoRec 整理的数据，也借用了它的两个训练思路。HaRS 用物品周围的邻居相似度来估计样本难度，再按难度调 DPO 的力度。NoDO 会在 LoRA 路径加一点噪声。我们把这两步接到自己的 CLIP 召回和逐条打分流程里，最后看自然候选上的 HR 和 NDCG。
>
> 模拟 Top-50 测试里，难负例 HaNoRec 的 HR@10 是 **3.62%**，难负例 DPO 是 **3.58%**，只差 **0.04 个百分点**。两组的 NDCG@10 四舍五入都是 **1.95%**。这个比较用的是我们的两阶段流程，原论文的小候选池指标用的是另一套评估口径。

[HaRS 实现](mm-retrieval-rerank-fork/src/mmrr/hars.py) · [NoDO 实现](mm-retrieval-rerank-fork/src/mmrr/models/preference.py) · [模拟结果总表](mm-retrieval-rerank-fork/README.md#end-to-end-study)

## 评估、结果和图像遮蔽

### 20. Recall@K、HR@k 和 NDCG@k 怎么算？为什么需要两类排名指标？

> 每个用户只有一个正确答案。Recall@K 问的是，第一步找的 K 个候选里有没有它。HR@k 问的是，排完以后它进没进前 k。NDCG@k 还关心它到底排第几。
>
> 比如都进了前十，第一名和第十名在 HR@10 里都算一次命中。在 NDCG@10 里，第一名得 1 分，第十名大约得 0.29 分。目标没被召回，后面两项都记零。这样既能看有多少人进了前十，也能看目标有没有往前挪。

**白板公式**（每个用户一个目标，`N` 是用户数）

~~~text
Recall@K = (Σ_u 1[target_u ∈ C_K(u)]) / N
HR@k     = (Σ_u 1[rank_u ≤ k]) / N
NDCG@k   = (Σ_u 1[rank_u ≤ k] / log₂(rank_u + 1)) / N
~~~

[评估实现](mm-retrieval-rerank-fork/src/mmrr/eval/experiment.py)

### 21. 为什么只跑了 499 个用户的 Qwen，却能报告 5,000 人的指标？

> 因为这次验证里，5,000 人中只有 499 人的目标进了 Top-100。剩下 4,501 人，怎么排都排不到答案，指标已经确定是零。所以为了算离线指标，只需要给这 499 人打分。
>
> Qwen 最后把 122 个目标排进了前十。问“召回命中的人里排得怎么样”，就算 **122/499 = 24.45%**。问“整个系统对所有人怎么样”，就算 **122/5,000 = 2.44%**。上线时我们事先不知道谁会命中，测延迟也要做完整请求。

[自然 Top-100 验证归档](mm-retrieval-rerank-fork/reports/rerank/interest100-rerank-report.md)

### 22. 已经跑出了什么结论？Qwen 为什么降分，你怎么验证原因？

> 已有的 Top-100 验证里，候选一条都没换，CLIP 原排序的 HR@10 是 **3.02%**，zero-shot Qwen 重排后是 **2.44%**。我的猜测是，Qwen 能看懂两条视频像不像，但下一次会不会看，还需要从用户行为里学。
>
> 验证这个猜测，我会把候选锁死，再换排序方式。报告里的 Top-50 模拟测试就是按这个思路比的。CLIP 是 **2.86%**，zero-shot 是 **2.32%**，加难负例做 SFT 后是 **3.24%**，难负例 DPO 是 **3.58%**。这个对照在模拟里体现了任务训练的收益。再细看提示词，DPO 换成 `minionerec` 模板后是 **3.42%**，也能看出输入怎么写会影响排序。

[Top-100 验证归档](mm-retrieval-rerank-fork/reports/rerank/interest100-rerank-report.md) · [模拟结果总表](mm-retrieval-rerank-fork/README.md#end-to-end-study)

### 23. 为什么要人为塞入正确答案？它说明了什么？

> 我们想知道，第一步没找着答案的那些用户，如果硬把答案放到候选里，Qwen 能不能认出来。做法是从漏召回的用户里挑 500 个，每人用真实目标替换原来最后一个候选，再把它随机插到列表里。
>
> 验证归档里，随机插入后有 **8.2%** 的目标在前十，zero-shot Qwen 重排后是 **9.4%**，均匀随机的期望是 **10%**。在报告的模拟对照里，随机插入是 **9.80%**，zero-shot 是 **9.60%**，难负例 DPO 是 **10.80%**。两组数字都在说这 500 个特意选出来的难样本，端到端还是看自然候选上的 HR。

[人工注入验证归档](mm-retrieval-rerank-fork/reports/rerank/interest100-rerank-report.md) · [模拟注入对照](mm-retrieval-rerank-fork/README.md#target-injection-diagnostic)

### 24. CLIP 和 Qwen 怎么融合？融合权重是多少？

> 两边的分数单位不一样，所以我直接融名次。先把两边的名次换成 0 到 1 的相对位置，再用加权后的名次重新排序。报告的模拟例子里，CLIP 占 **75%**，Qwen 占 **25%**，测试 HR@10 从 CLIP 原顺序的 **2.86%** 到了 **3.02%**。
>
> 权重是用验证集 NDCG@10 选的，候选值是 0、0.25、0.5、0.75 和 1。用已有的 Top-100 zero-shot 验证分数单独复算时，选到的是 **0**，也就是完全沿用 CLIP 的顺序。

[融合实现](mm-retrieval-rerank-fork/src/mmrr/eval/experiment.py) · [模拟融合结果](mm-retrieval-rerank-fork/README.md#end-to-end-study) · [验证集融合复算](mm-retrieval-rerank/outputs/pipeline_acceptance_1c557c9977884b61a18e3c027b01d121/fusion.meta.json)

### 25. 你怎么做图像遮蔽？能得出怎样的结论？

> 我用同一组用户、同一份候选、同一个 DPO 模型跑两遍。历史和标题都一样，唯一改动就是第二遍把候选图从 Qwen 输入里拿掉。CLIP 召回早就做好了，所以两组看的还是同一批候选。
>
> 模拟测试里，拿掉图以后，HR@10 从 **3.58% 降到 3.36%**，NDCG@10 从 **1.95% 降到 1.81%**，Recall@50 两边都是 **6.88%**。也就是说，这个差异出在 Qwen 怎么排这批候选。保留图片的 HR@10 高 **0.22 个百分点**，模拟区间跨过了零，所以我会说这组数字里有图更好，提升幅度还有不确定性。

[评分配置](mm-retrieval-rerank-fork/src/mmrr/score.py) · [模拟图像遮蔽结果](mm-retrieval-rerank-fork/README.md#image-ablation-negative-difficulty-and-candidate-budgets)

### 26. 面试官指着简历问“DPO 提升了多少”或指着 README 的 3.58% 追问，你怎么回答？

> 按 README 里的模拟测试，难负例 DPO 的端到端 HR@10 是 **3.58%**。和 CLIP 的 **2.86%** 比，高 **0.72 个百分点**。和从同一个 SFT 模型起步、用随机负例的 DPO 比，高 **0.24 个百分点**。
>
> 这个 3.58% 是用全部 5,000 人算的。第一步找到了 344 人的目标，第二步把其中 179 个排进前十，所以是 **179/5,000**。如果只问找到了目标的这批人排得怎么样，才是 **179/344，也就是 52.03%**。

[README 模拟主结果](mm-retrieval-rerank-fork/README.md#end-to-end-study)

### 27. 样本方差、置信区间和显著性怎么做？

> 两个方法都在同一批用户上评，我会先算每个用户的分数差。比如只有 DPO 把这条排进前十，差就是 +1，只有 CLIP 命中就是 −1，两边一样就是 0。然后做 2,000 轮重采样。每轮有放回地抽同样多的用户，算一次平均差，取中间 95% 当配对区间。换训练种子再算标准差，可以看换一次训练结果会飘多少。
>
> 报告里的模拟区间用配对命中数做正态近似。难负例 DPO 比 CLIP 高 **0.72 个百分点**，区间是 **[+0.33, +1.11]**。难负例 DPO 比随机负例 DPO 高 **0.24 个百分点**，区间是 **[−0.12, +0.60]**，跨了零，所以这组模拟数字体现的是提升趋势。难负例 DPO 的模拟 Test HR@10 跨种子标准差是 **0.14 个百分点**。

[配对评估实现](mm-retrieval-rerank-fork/src/mmrr/eval/experiment.py) · [模拟结果与配对区间](mm-retrieval-rerank-fork/README.md#between-seed-variation-and-paired-differences)

## 工程、调试和取舍

### 28. 怎么复现结果？长时间的 GPU 任务断了怎么办？

> 想复算已经归档的验证指标，直接把固定的 Top-100 候选和保存的逐用户分数交给 `pipeline.py evaluate`，CPU 就能跑。像报告里的模拟 **3.58%**，则可以拿前十命中的人数 **179** 除以 **5,000**，对上表里的数字。
>
> 如果训练中途断了，用原配置从最新的完整 checkpoint 继续，接上 adapter、优化器和训练状态。如果打分断了，就用原命令和原输出目录重启。分数按用户分片保存，程序会检查候选和配置，再跳过已经写好的分片。

[复现与恢复手册](mm-retrieval-rerank-fork/docs/final_pipeline.md) · [验证分数归档](mm-retrieval-rerank-fork/reports/rerank/interest100_hits/val.jsonl)

### 29. 你会如何测延迟、吞吐和显存？最大的瓶颈在哪里？

> 我会先预热 GPU，用同一张卡和同一批完整请求，把召回、Qwen 重排、端到端分别计时。P50 就是中位耗时，P95 是 95% 的请求能在这段时间内完成。吞吐看每秒完成多少请求，显存看 GPU 的峰值分配。
>
> 报告按单张 4090、串行请求模拟。Top-50 时，召回 P95 是 **13.6 毫秒**，重排 P95 是 **5.58 秒**。改成 Top-100，重排 P95 是 **10.96 秒**，吞吐从 **0.217 降到 0.112 请求/秒**，峰值显存从 **10.4 到 10.5 GiB**。主要时间花在了 Qwen 逐条处理候选上。

[模拟延迟与资源数据](mm-retrieval-rerank-fork/README.md#latency-and-memory) · [测速实现](mm-retrieval-rerank-fork/src/mmrr/score.py)

### 30. 如果只有一天继续做，你先做什么？如果上线呢？

> 只有一天，我会先看召回漏掉的是谁。验证集 Top-50 只找回了 **357/5,000** 个目标。模拟测试按目标热度分组，训练里一次都没出现过的视频召回只有 **3.55%**，出现超过 100 次的有 **9.17%**。我会先用同一个 Top-50 预算看这个差距，再拿带用户行为的召回方法做个对照。
>
> 如果上线，我会先按延迟预算定 K。比如模拟里从 50 加到 100，HR@10 只多 **0.14 个百分点**，重排 P95 却接近翻倍。我会先从小候选池开始、把 Qwen 打分做成批处理，超时就用 CLIP 原排序，再盯实际的请求延迟和用户行为。

[召回验证对照](mm-retrieval-rerank-fork/reports/retrieval/history-interest-report.md) · [模拟分层](mm-retrieval-rerank-fork/README.md#recall-by-target-frequency) · [模拟延迟](mm-retrieval-rerank-fork/README.md#latency-and-memory)

### 31. Recall@50 只有 7.14%，你如何判断这个召回做得怎么样？

> 我会看它比基线好多少，也看它给后面留了多少空间。随机从 19,220 条里抽 50 条，命中率大概 **0.26%**。我们的验证集上，按流行度推荐是 **1.50%**，图文召回加上历史兴趣后到 **7.14%**，说明这些信号有用。
>
> 但这也意味着，哪怕 Qwen 把所有召回命中的目标都排到第一，端到端 HR@10 也最多 **7.14%**。报告的模拟分组里，训练中没出现过的目标只有 **3.55%** 被召回，热门组是 **9.17%**。所以我会先看低频视频丢在哪，再和学用户行为的召回模型比一遍。

[期中报告](<Project Midterm Report.pdf>) · [召回验证报告](mm-retrieval-rerank-fork/reports/retrieval/history-interest-report.md) · [模拟分层](mm-retrieval-rerank-fork/README.md#recall-by-target-frequency)

## 面试前最后看一遍的数字

| 问到什么 | 记这组数字 | 口径 |
|---|---|---|
| 召回怎么改 | **6.48% → 7.14%**，324 → 357 个目标 | 验证实测，Top-50 |
| zero-shot 发生了什么 | **3.02% → 2.44%**，151 → 122 个前十命中 | 验证实测，Top-100，端到端 HR@10 |
| DPO 提升多少 | **2.86% → 3.58%**，对 CLIP 高 **0.72 个百分点** | 模拟测试，Top-50，端到端 HR@10 |
| 难负例有什么用 | 随机 DPO **3.34%**，难负例 DPO **3.58%** | 模拟测试，Top-50，同一 SFT 起点 |
| 拿掉候选图会怎样 | HR@10 **3.58% → 3.36%**，NDCG@10 **1.95% → 1.81%** | 模拟测试，Top-50，召回固定 **6.88%** |
| 扩到 100 值不值 | 召回 **6.88% → 9.56%**，DPO HR@10 **3.58% → 3.72%**，重排 P95 **5.58 → 10.96 秒** | 模拟测试，Top-50 对 Top-100 |

口述算式可以记成 **344 人找到了，179 人排进前十**。模拟测试的条件 HR@10 是 **179/344 = 52.03%**，端到端 HR@10 是 **179/5,000 = 3.58%**。

[验证集召回报告](mm-retrieval-rerank-fork/reports/retrieval/history-interest-report.md) · [验证集重排报告](mm-retrieval-rerank-fork/reports/rerank/interest100-rerank-report.md) · [README 模拟结果](mm-retrieval-rerank-fork/README.md#end-to-end-study)

完整模拟报告可在项目根目录用 `git -C mm-retrieval-rerank-fork show 27e2440:reports/final_results_simulated.md` 查看。
