# 使用指南

本项目提供视觉吃水重构、CCTV–AIS 关联、排放计算与验证统计代码。观测数据、模型权重、相机标定和实验配置由使用者提供。

## 安装

使用 Python 3.10 或更新版本：

~~~bash
python -m pip install .
python -m vesselviz --help
~~~

本地 CLIPSeg 推理使用可选依赖：

~~~bash
python -m pip install ".[vision]"
~~~

SAM2 按官方仓库说明单独安装。仅使用已有结果绘图或计算排放时，无需视觉模型依赖。

## API 与配置

复制所需的 example.json 为 local.json，填写输入路径和参数。

reconstruction 配置中的 vlm 部分填写完整 endpoint、模型名称、任务提示词文件、生成参数和密钥环境变量名。接口采用多模态 chat-completions 格式，可连接对应格式的 Qwen 服务。密钥从环境变量读取。

CLIPSeg 指定本地模型或模型标识，SAM2 指定架构配置和权重路径。CLIPSeg 默认只读取本地文件，下载需要显式修改 local_files_only。

homography 是图像到局部米制水面的映射；projection_matrix 是三维世界坐标到图像的 3×4 矩阵。二者必须共用坐标基准，且匹配图像预处理方式。

## 执行顺序

1. matching：提供跟踪观测与局部米制 AIS，生成轨迹级 MMSI。
2. prepare_observations：连接已接受匹配与船舶型深档案。
3. reconstruction：输入连接后的 observations.json，输出帧级干舷和 passage 级视觉吃水。
4. prepare_inventory：将视觉吃水连接至明确积分时长的活动记录。
5. emissions：填入排放模型系数、低负荷表及来源，生成清单。
6. validation：提供核验吃水、排放结果、长宽量测或人工关联标签，生成验证统计。

~~~bash
python -m vesselviz match --config configs/matching.local.json --output results/matching
python -m vesselviz prepare --config configs/prepare_observations.local.json --output results/linked_observations
python -m vesselviz reconstruct --config configs/reconstruction.local.json --output results/reconstruction
python -m vesselviz prepare --config configs/prepare_inventory.local.json --output results/inventory_inputs
python -m vesselviz emissions --config configs/emissions.local.json --output results/emissions
python -m vesselviz validate --config configs/validation.local.json --output results/validation
~~~

把各阶段实际输出路径填入下一阶段配置。每次运行使用新输出目录。字段定义见 DATA_SCHEMA.md。

传统吃水输入使用 seim_draft_m，视觉输入使用 visual_draft_m。型深和上边界必须使用相同的垂直定义，观测中通过 upper_reference_matches_moulded_depth 显式确认。

参考排放使用同一模型、速度、功率、排放因子和时间覆盖，只替换吃水。低负荷修正规则相同，数值随各自负荷重新计算。

## 出图与选图

~~~bash
python -m vesselviz association --config configs/association.local.json --output results/association_figures
python -m vesselviz hull --config configs/hull.local.json --output results/hull_figures
python -m vesselviz pick --kind association --config configs/association.local.json --output results/selected_frame
~~~

association 输出统一检测框、真实 Track ID 与两行信息卡。hull 读取已有整船 mask、CLIPSeg 数组和船体多边形生成五列图。弹窗选图要求存在该原图的唯一标签记录。图集 gallery.html 可导出选择结果，再用 assemble 拼图。

重构输出的干舷边界和 mask 使用原图坐标；五列展示的整船 mask 需要单独提供，不能用船体侧面 mask 替代。

## 检查

~~~bash
python -m unittest discover -s tests -v
python scripts/check_release.py
~~~

测试使用程序样例和解析几何数据。外部 API 与模型推理需要在配置后自行验证。
