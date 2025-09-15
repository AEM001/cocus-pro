# SAM分割系统运行说明

## 运行命令

```bash
# 激活环境并运行
conda activate camo-vlm
cd /home/albert/code/CV
python refactor_sam_sculpt.py
```

## 修改样本

编辑脚本第259行：
```python
sample_name = "f"  # 改为 "dog" 或 "q"
```

## 调整象限框大小

编辑脚本第155行（`create_quadrant_visualization`函数）：
```python
ratio: float = 0.35  # 增大这个值，例如改为 0.5 或 0.6
```

结果保存在 `outputs/refactor_sculpt/样本名/`
