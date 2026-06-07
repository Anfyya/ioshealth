# ioshealth

SwiftUI iOS App 工程。目标是端侧健康异常检测：

- 首次授权后读取 Apple Health / HealthKit 历史数据
- 聚合为 4 小时数据块，做缺失处理和训练集标准化
- 本地训练个人重建模型
- 内置 PMData16 4h、PMData16 lifted 8维 4h、RAIS57 日级多人预测先验，并用用户历史数据本地微调个人预测模型
- 用个人重建、个人预测、多人预测三路分位数融合生成异常等级
- 用户模型保存在本机 Application Support，不上传服务器

## 打开方式

在 macOS 用 Xcode 打开：

```bash
open ioshealth.xcodeproj
```

需要真机运行并开启 HealthKit capability。Bundle ID 当前是 `com.codex.healthanomaly`，上线前需要改成你的开发者账号可用的 ID。

## 本地静态校验

当前 Windows 环境不能执行 Xcode 编译，可先运行无第三方依赖的结构校验：

```bash
python tools/validate_ioshealth.py
```

通过后仍需要在 macOS/Xcode 上做真实编译、签名和真机 HealthKit 权限验证。

## GitHub Actions 自动编译 IPA

仓库内置工作流 [.github/workflows/build-ipa.yml](.github/workflows/build-ipa.yml)，在 macOS runner 上用 `xcodebuild` 自动编译。触发条件：

- push 到 `main` 分支
- 推送 `v*` 版本标签（会额外把 IPA 附到 GitHub Release）
- Actions 页面手动 `Run workflow`

产物是**未签名**的 `ioshealth.ipa`，可在对应运行记录的 **Artifacts** 里下载（`ioshealth-unsigned-ipa`）。因为没有配置 Apple 开发者证书，CI 用 `CODE_SIGNING_ALLOWED=NO` 跳过签名，所以这个 IPA 不能直接安装，需要用会重新签名的侧载工具：

- **AltStore / SideStore**、**Sideloadly**（用免费 Apple ID 重签，HealthKit 等权限在重签时随 entitlements 写入）
- **TrollStore**（受支持的系统版本可永久安装）

如果以后要让 CI 直接产出**已签名、可分发**的 IPA，把 Apple 签名证书（`.p12`）和描述文件作为仓库 Secrets 配置，再改用 `xcodebuild archive` + `-exportArchive` 走真实签名流程即可。

