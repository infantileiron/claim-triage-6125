# 禾序 · 报销异常分类复核

财务共享组先校准历史人工标签的可靠性，再训练可解释的逻辑回归版本，按异常概率排定待核票据；最终判断和签核由人员完成。原生 HTML/CSS/JS 五个业务页面：样本池、标注校准、训练配置、分类评估、风险复核。

## 真实模拟场景

模拟客户禾序咨询有 180 名顾问，财务共享组 6 人每周核对差旅报销。原流程是导出差旅明细、在工作簿标记疑点、随机抽取票据、在群里确认处理。三个痛点是复核顺序依赖个人经验，模型指标和业务阈值混为一谈，最终判断难以追溯到当时的样本和模型。上游输入报销金额、出行里程、节假日、人工历史标签，以及多位财务各自的独立意见与少量已确认金标准；本项目替代样本整理、标注可靠性校准、风险排序、版本评估与复核记录；下游由财务负责人结合原票据作最终处理。本项目不连接真实财务系统，也不自动拒付。

| 角色 | 主要页面 | 权限 | 典型动作 |
|---|---|---|---|
| admin | 全部、金标准、签核 | 读写、确认金标准、训练、复核、签核和审计 | 确认金标准并签核已填人工判断的任务 |
| analyst | 样本、校准、训练、评估、复核 | 读写业务、录入意见和提交校准/训练，不能确认金标准或签核 | 录入独立意见并生成复核候选 |
| viewer | 五页只读 | 可看与下载，不能写入、校准或训练 | 查看校准依据、模型和冻结评分 |

## 一日剧本

1. 08:50 viewer 查看 `HX-1115-001` 等历史样本，初始 300 笔、260 笔有人工标签、40 笔待标注；种子还会写入多位财务的不完整重叠意见与 6 笔金标准。
2. 09:00 analyst 录入 `HX-QA-NEW`：金额 2600 元、里程 180 km、节假日“是”、标签“待标注”。
3. 09:10 在标注校准页为票据录入各财务的独立意见；同一位财务对同一票据只保留最新一版，并追加修订记录。
4. 09:15 admin 对少量已核对票据确认金标准。金标准只用于锚定方向，不参与留出评估。
5. 09:20 创建“11月标注校准”，worker 用不完整重叠意见 + 金标准估计各人漏报/误报率与逐票后验。
6. 09:30 查看校准依据：各财务漏报率、误报率、区间与可信/反向状态；逐票给出后验区间，宽区间不给软标签。
7. 09:40 创建“11月差旅风险基线”，复核阈值 0.5，可选使用校准后的软标签；worker 执行固定 seed 的分层留出训练。
8. 09:50 查看 Precision/Recall/F1、混淆矩阵及 0.3/0.5/0.7 阈值敏感度，以及使用了哪些软标签/金标准。
9. 10:00 生成当前版本高风险待标注样本的复核任务，重复生成不会重复添加。
10. 10:10 analyst 复核原票据并填“核验正常”或“需进一步核实”及理由。
11. 10:20 admin 签核人工结论；10:30 viewer 分别下载 `annotation-calibration-<批次ID>.json` 与 `claim-triage-<版本ID>.json` 留档。

## 算法与口径

输入金额（0–100000元）、里程（0–10000km）、节假日布尔与人工标签 0/1/空。派生特征“每公里金额”为 amount/(distance+1)，加一避免零里程除零。标签不进入特征矩阵。

### 标注可靠性校准

历史人工标签可能把多人共同误判当成真相，因此新增独立的校准层。模型为二分类 Dawid–Skene 变体：票据真实状态 z_i 未知，`P(y_ij=1|z_i=1)=sensitivity_j`，`P(y_ij=0|z_i=0)=specificity_j`，漏报率=1−灵敏度、误报率=1−特异度。用 EM 在**不完整重叠**的独立意见上求解，少量已确认金标准的 z 固定为确认值以锚定方向；Beta(1,1) 平滑避免小样本 0/1 极端值。多数票只作为对照展示，**不当作真值**。

证据门槛（任一不满足即整批拒绝并保存原因，不输出虚假高置信度）：独立意见 ≥12 条、标注者 ≥2 人、意见数 ≥4 的标注者 ≥2 人、带意见的金标准 ≥2 笔、且至少一个人群连通分量含 ≥2 笔金标准锚点。意见人群按“是否有人共同评过同一票据”划分连通分量；缺少金标准锚点的人群不参与参数估计，其票据概率留空并说明原因。每位标注者另按意见数、与他人/金标准重叠数判定 insufficient/unanchored；灵敏度与特异度同时显著低于随机判为 reversed（反向录入），区间跨过随机水平判为 unstable。

逐票后验用参数自助法（seed 3106+7，40 次重抽）给出 5%–95% 区间。区间宽度 ≤0.4 记为 estimated 可作软标签；≤0.7 记为 uncertain 仅参考；>0.7 记为 insufficient，点估计一并隐去。证据不足时 `estimate()` 抛 `CalibrationError`，批次进入 failed 并保存 `reasons`/`checks`，页面逐条解释。

校准批次状态为 queued→running→done/failed。提交时冻结全部意见、金标准与 sha256 快照哈希；失败重试沿用同一份快照（`attempts` 累加，意见不重复累计），启动恢复也会把 queued/running 重新入队但不会重新读取意见。意见更正采用“当前一版 + 追加修订记录”，同一(票据, 财务)只有一版；提交训练时取校准批次的完整一版，之后的更正与重新校准都不改变既有训练版本。

### 训练与评估

已标注样本至少40笔且每类至少10笔，按标签分层分成75%训练/25%留出（seed=3106）。StandardScaler **只在训练集拟合**；Pipeline 后接 L2 LogisticRegression（C=1，max_iter=1000）。概率为 sigmoid(w·标准化特征+b)，阈值以上预测异常。Precision=TP/(TP+FP)，Recall=TP/(TP+FN)，F1=2PR/(P+R)，分母为零时指标为0。混淆矩阵合计等于留出集样本量。每个版本保留完整输入快照、训练/测试编号、训练集标准化均值、系数、留出概率与待标注概率、sha256 checksum。

使用软标签时：estimated 票据的训练目标改写成两条加权样本（概率 p 与 1−p，权重随 |2p−1| 和区间宽度衰减），等价于加权交叉熵；留出集仍只保留原始人工标签，金标准与软标签都不进入留出评估。版本记录使用的校准批次、快照哈希、软标签清单与金标准训练清单，并报告留出集中软标签/金标准重叠数（应为 0）。

阈值敏感度在同一留出集计算。反复选择阈值后，该集合不能继续当成最终独立验收集。本项目用于小规模业务演示，不包含额外生产验收集。标准化系数可解释模型关联，不作为因果结论。类别不足会以 failed 状态保存明确错误；重复编号返回409，无效金额/标签返回422。输入修订后须新建训练版本，旧快照不变。任务只对其快照里的待标注样本评分，不提供模型在线部署端点。评分仅排定人工复核顺序。

## 快速启动

```bash
cp .env.example .env
docker compose config --quiet
docker compose up -d --build
# 自动 seed，也可显式执行两次验证
 docker compose exec -T backend python -m scripts.seed_demo
```

前端 http://localhost:18205 ，后端 http://localhost:19205/docs ，MySQL 映射17705。停止并删除演示数据：`docker compose down -v --remove-orphans`。Compose 顶层 name=claim-triage，五个服务各有健康检查；backend/worker 等待db+Redis健康，frontend等待backend健康。数据库、Redis队列、worker产物均有独立命名卷。worker单进程，启动时恢复 queued/running 版本；不承诺多worker exactly-once。

账号为 `admin@demo.local` / `analyst@demo.local` / `viewer@demo.local`，密码均 `Demo123!`，JWT 8小时。密码PBKDF2使用每账号随机盐；JWT_SECRET为演示环境值。按角色后端强制检查，前端只读账号按钮禁用。

## 技术栈和目录

| 层 | 实现 |
|---|---|
| UI | 原生 ES modules、HTML、CSS；Noto Serif SC/Noto Sans SC 双字体 |
| API | Python3.11、FastAPI、Pydantic、SQLAlchemy、PyJWT |
| 算法 | scikit-learn1.6.0、NumPy2.2.0；Dawid–Skene 标注校准 |
| 异步 | Redis列表队列、独立Python worker与心跳 |
| 数据 | MySQL8.0、Redis7；Nginx反代 `/api/` 去前缀 |

`frontend/src` 下 api/stores/types/components/common/hooks/pages/router/utils/constants 分离网络、状态、契约、组件、表单、页面和枚举；标注校准页为 `pages/calibration.js`。`backend/src` 下 routes/controllers/services/models/schemas/middlewares/utils/constants/config 分离HTTP、查询、算法、持久化、校验、授权和配置。`backend/scripts/seed_demo.py` 用自然键只插入缺失账号/报销/意见，不覆盖人工修订；`backend/src/services/calibration.py` 为纯算法模块（无数据库依赖）。`backend/migrations/001_initial.sql` 记录初始架构，实际初始DDL由SQLAlchemy metadata建立。

共享枚举 ROLES/STATES/DECISIONS/LABELS 及校准状态 ANNOTATOR_STATUSES/CLAIM_STATUSES 位于后端 `backend/src/constants/enums.py` 和前端 `frontend/src/constants/enums.js`；Pydantic枚举校验位于 `backend/src/schemas/business.py`，权限检查位于 `backend/src/middlewares/auth.py`。状态徽章/表格/表单字段为共享组件，两种中间件提供request ID和安全响应头；全局异常处理与审计记录横切业务。

本地开发：启动Compose的db/redis，将DATABASE_URL和REDIS_URL改指宿主映射；Python3.11虚拟环境安装 backend/requirements.txt，在backend目录 `uvicorn src.main:app --reload`。前端建议继续用Compose Nginx以保留同源API代理，也可使用可代理 `/api` 的本地静态服务器。单独开放静态文件不能替代API代理。

环境变量：COMPOSE_PROJECT_NAME、FRONTEND_PORT、BACKEND_PORT、DB_PORT、DB_PASSWORD、JWT_SECRET、SEED_DEMO、NETWORK_SUBNET，均见.env.example。NETWORK_SUBNET默认10.253.106.0/24，可改为本机未用子网。

## 已知范围

逐条表单/API录入和完整字段修订；无CSV/Excel导入、删除、分页或外部审批通知。原始seed为确定性模拟数据，不能推断真实风险；其中意见与金标准也是模拟，仅用于演示校准。校准模型假设各财务在给定真实状态后独立判断，无法识别所有人一致的共同偏差——这正是保留金标准与逐票区间的原因；若整批意见都无金标准锚点会拒绝给出概率。软标签是估计值，不能替代人工确认，也不自动回写训练标签；风险复核任务同样须人工在样本页确认标签后重新训练。Google Fonts联网加载，离线字体回退。测试在 `backend/tests/`（`python -m pytest tests`）；截图和报告在源码外 `test-artifacts/gb-3106/`。

License：MIT，见 LICENSE。
