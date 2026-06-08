# ioshealth

端侧(on-device)健康异常检测 iOS App(SwiftUI,iOS 26)。全部在你的 iPhone 上运行,数据不上传。

## 它做什么

- 首次授权后默认读取 Apple Health 可返回的**全部历史**,也可手动选择最近 5 年、3 年或 7 个月;读取 8 个指标:心率、步数、活动消耗、心率变异性(HRV)、血氧、呼吸频率、睡眠、运动时长
- 聚合为 4 小时数据块,做缺失处理 + 按个人数据 z-score 标准化,切成 252 块(约 42 天)的滑动窗口
- **双模型异常检测**,基于 Anomaly Transformer(ICLR 2022),用 [MLX](https://github.com/ml-explore/mlx-swift) 在端侧运行:
  - **个人重建模型**:用你自己的数据**在 iPhone 上训练**(MLX + AdamW + masked MSE),训练轮数按窗口数动态提高到 24-64 轮。擅长发现"多个指标之间的配合关系被打破"
  - **多人预测基座**:内置预训练权重(16 人数据,cross-attention 预测解码),只推理。擅长发现单指标突变 / 渐变趋势 / 节律偏离
  - 两类异常互补(经合成数据基准验证):重建擅长联动破裂与突发,预测擅长单变量与渐变
- 原始样本先经过分钟级哨兵规则,例如心率 200 持续 5 分钟会按原始开始/结束时间直接列为异常段,不会被 4 小时模型桶平均掉
- 两路误差按**你自己的分布**做分位校准后融合,只展示达到预警线的异常时间段,并标明来源是个人模型、通用基座、双模型一致或原始数据哨兵
- App 主界面使用底部 Tab:概览、异常、数据;不会再用一串绿色窗口冒充分析结果
- 结果保存在本机 Application Support,不上传服务器

## 模型与依赖

- 端侧 ML 运行时:`mlx-swift`(Swift Package,首次构建自动拉取)
- 预测基座权重 [ioshealth/Resources/PredictionBase.safetensors](ioshealth/Resources/PredictionBase.safetensors) 由配套的 PyTorch 训练工程导出(Anomaly Transformer 重建/预测);位置编码、距离矩阵等确定性 buffer 在端侧按公式重算,不入包。

## 打开方式

在 macOS 用 Xcode 打开：

```bash
open ioshealth.xcodeproj
```

首次构建会自动拉取 `mlx-swift`(需要联网)。真机运行需开启 HealthKit capability。Bundle ID 当前是 `com.codex.healthanomaly`，上线前需要改成你的开发者账号可用的 ID。

## GitHub Actions 自动编译 IPA

仓库内置工作流 [.github/workflows/build-ipa.yml](.github/workflows/build-ipa.yml)，在 macOS runner 上用 `xcodebuild` 自动编译。触发条件：

- push 到 `main` 分支
- 推送 `v*` 版本标签（会额外把 IPA 附到 GitHub Release）
- Actions 页面手动 `Run workflow`

产物是**未签名**的 `ioshealth.ipa`，可在对应运行记录的 **Artifacts** 里下载（`ioshealth-unsigned-ipa`）。因为没有配置 Apple 开发者证书，CI 用 `CODE_SIGNING_ALLOWED=NO` 跳过签名，所以这个 IPA 不能直接安装，需要用会重新签名的侧载工具：

- **AltStore / SideStore**、**Sideloadly**（用免费 Apple ID 重签，HealthKit 等权限在重签时随 entitlements 写入）
- **TrollStore**（受支持的系统版本可永久安装）

如果以后要让 CI 直接产出**已签名、可分发**的 IPA，把 Apple 签名证书（`.p12`）和描述文件作为仓库 Secrets 配置，再改用 `xcodebuild archive` + `-exportArchive` 走真实签名流程即可。
