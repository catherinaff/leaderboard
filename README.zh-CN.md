# IoTBench 榜单与复现实验代码

[在线榜单](https://catherinaff.github.io/leaderboard/) · [完整运行说明](reproduction/README.md) · [已完成实测表](reproduction/docs/COMPLETED_RESULTS_20261124.md) · [逐方法实现审查](reproduction/docs/IMPLEMENTATION_REVIEW.md)

当前榜单使用 UCI-HAR（6类）、PAMAP2（12类）、USC-HAD（12类）、RealWorld（8类）。总榜对四项 Accuracy 等权平均；实测优先，缺项保留论文值。每个计入的种子都必须覆盖全部测试集／受试者，未满五种子标为暂定。

最近一次结果核验为 **2026-11-24**：59 组实测，其中53组完成五种子、2组暂定、4组零样本，共2,517条核验通过的折记录。历史快照和审查证据继续保留。

`reproduction/` 提供当前19种方法的代码／接口／未运行状态说明，以及4种历史方法的实现。MOSAIC目前只有论文数据；HARGPT和ByMyEyes只接好接口，没有付费实跑。oneHAR、TimeLLM、SensorLLM等的适配差异均单独说明，不能将全部适配实验统称为严格论文复现。

固定监督种子为 **42、123、2021、2024、3407**。常用训练上限200轮，验证集早停patience25、min_delta0.0005，梯度裁剪1；plateau学习率调度为减半、patience8、最低1e-6。oneHAR适配版上限80轮，SensorLLM上限30轮、早停10。OneLLM为一次冻结权重零样本评估，没有五次独立训练。

各方法实际优化器、初始学习率、batch size、预训练阶段、监控指标及实际停止轮次见[外层README参数表](README.md#executed-training-settings)和[逐单元格训练配置](reproduction/configs/training_recipes.json)。这些参数来自保存的运行记录，不能直接当作各原论文的训练参数。

数据使用训练折拟合的中位数填补和均值／标准差归一化。UCI保留官方测试集，其余三套数据覆盖全部受试者的LOSO；每折另留一个验证受试者。各模型使用的传感器子集、窗口适配和预训练权重差异见对应方法README。

审查证据包括官方固定版本25个关键文件的重新下载哈希比对、移植运算数值／梯度检查、原模型前后向测试、预测／权重／划分／源代码快照核验。原论文数据协议的完整复现与准确率校准仍需分别确认，不能仅凭代码哈希或分数接近认定完成。

代码包不包含原始数据、模型权重、论文PDF、密码、私钥或真实API配置。依赖的原仓库由固定版本下载脚本获取并遵守各自许可；现有网页、任务页、数据集下载页和上传入口保留。
