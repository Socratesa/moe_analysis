import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import BitsAndBytesConfig
from datasets import load_dataset
from tqdm import tqdm
import os
import logging

# ================= ⚙️ 用户配置区域 =================

# 1. 模型 ID
MODEL_ID = "/home/lhd/Qwen/Qwen3-30B-A3B-Instruct-2507"

# 2. 分析模式开关
#   - 设为整数 (例如 10): 只分析第 10 层 (速度最快，适合调试)
#   - 设为 None: 分析所有 MoE 层 (适合全量分析，生成每一层的热力图)
TARGET_LAYER = 30
ENABLE_COUNT_FILTER = True  # 开关：True 表示开启限制，False 表示不限制
MIN_COUNT_THRESHOLD = 50    # 阈值：限制的最小次数

# 3. 数据配置
OUTPUT_DIR = "moe_analysis_report"
NUM_SAMPLES = 100        # 采样数量 (样本越多越准)
MAX_SEQ_LEN = 1024       # 序列长度
BATCH_SIZE = 4           # 适当增大 Batch 可加速推理
NUM_COACTIVATORS = 10    # 每个专家 top co-activators 数量
NUM_TOP_ACTIVE = 20      # 打印热门专家数量  

# ===================================================

# 设置日志和目录
os.makedirs(OUTPUT_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MoEContextAnalyzer:
    def __init__(self, model_id, output_dir):
        self.output_dir = output_dir
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 1. 加载模型
        logger.info(f"🚀 Loading model: {model_id}...")
        try:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            self.model = AutoModelForCausalLM.from_pretrained(
                model_id,
                quantization_config=bnb_config,
                torch_dtype="auto",
                device_map="auto",
                trust_remote_code=True
            ).eval()
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            exit(1)

        # 2. 探测专家配置
        self._detect_config()
        
        # 3. 初始化统计存储容器
        # layer_stats[layer_idx] = { "co_matrix": Tensor, "counts": Tensor }
        self.layer_stats = {}
        self.hooks = []

    def _detect_config(self):
        """自动识别不同架构的 MoE 配置"""
        config = self.model.config
        if hasattr(config, "num_experts"):
            self.num_experts = config.num_experts
            self.top_k = config.num_experts_per_tok
        elif hasattr(config, "num_local_experts"): # Mixtral / Llama-MoE
            self.num_experts = config.num_local_experts
            self.top_k = config.num_experts_per_tok
        else:
            logger.warning("⚠️ Unknown MoE config. Defaulting to 64 experts, top-k=2.")
            self.num_experts = 64
            self.top_k = 2
        
        logger.info(f"⚙️ Config Detected: {self.num_experts} Experts, Top-{self.top_k} Routing")

    def _get_hook_fn(self, layer_idx):
        """生成高效的 Hook 函数"""
        def hook_fn(module, input, output):
            # output shape: [batch, seq_len, num_experts] (logits)
            # 展平 -> [total_tokens, num_experts]
            logits = output.view(-1, output.size(-1))
            
            # 提取 Top-K 索引
            # 只取 indices，不需要 gradients
            with torch.no_grad():
                _, indices = torch.topk(logits, k=self.top_k, dim=-1)
                
                # === ⚡️ 性能关键点：立即移至 CPU 计算 ===
                indices = indices.cpu()
                num_tokens = indices.shape[0]

                # 初始化该层统计器 (如果尚未初始化)
                if layer_idx not in self.layer_stats:
                    self.layer_stats[layer_idx] = {
                        "co_matrix": torch.zeros((self.num_experts, self.num_experts), dtype=torch.float64),
                        "counts": torch.zeros(self.num_experts, dtype=torch.float64)
                    }
                
                stats = self.layer_stats[layer_idx]

                # === ⚡️ 向量化计算 (Vectorized) ===
                # 构造 Multi-hot Mask [tokens, experts]
                mask = torch.zeros(num_tokens, self.num_experts, dtype=torch.float64)
                mask.scatter_(1, indices.to(torch.long), 1.0)
                
                # 1. 更新单个专家计数 (Sum columns)
                stats["counts"] += mask.sum(dim=0)
                
                # 2. 更新共现矩阵 (Matrix Multiply)
                # Mask.T [E, N] @ Mask [N, E] = [E, E]
                stats["co_matrix"] += torch.matmul(mask.t(), mask)
                
        return hook_fn

    def register_hooks(self, target_layer=None):
        """
        target_layer: None (所有层) 或 int (指定层)
        """
        logger.info("🔗 Registering hooks...")
        count = 0
        
        # 遍历模型层
        if hasattr(self.model, "model") and hasattr(self.model.model, "layers"):
            layers = self.model.model.layers
        else:
            logger.error("Could not find layers in model.")
            return

        for i, layer in enumerate(layers):
            # 如果指定了层，且当前层不是目标层，则跳过
            if target_layer is not None and i != target_layer:
                continue
            
            # 寻找 Gate 模块
            target_module = None
            if hasattr(layer, "mlp") and hasattr(layer.mlp, "gate"): # Qwen
                target_module = layer.mlp.gate
            elif hasattr(layer, "block_sparse_moe") and hasattr(layer.block_sparse_moe, "gate"): # Mixtral
                target_module = layer.block_sparse_moe.gate
            
            if target_module:
                h = target_module.register_forward_hook(self._get_hook_fn(i))
                self.hooks.append(h)
                count += 1
        
        if count == 0:
            logger.error(f"❌ No MoE layers hooked! (Target: {target_layer})")
        else:
            logger.info(f"✅ Hooked {count} layers.")

    def run_inference(self, num_samples, batch_size, seq_len):
        """加载数据并运行推理"""
        logger.info("📚 Preparing Dataset (WikiText)...")
        try:
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
            # 过滤短文本
            raw_texts = [x['text'] for x in ds if len(x['text']) > 200]
            texts = raw_texts[:num_samples]
        except Exception as e:
            logger.warning(f"Dataset load failed ({e}), using dummy data.")
            texts = ["AI scaling is fascinating. " * 50] * num_samples

        logger.info(f"🏃 Running Inference on {len(texts)} samples...")
        
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), batch_size), desc="Inferencing"):
                batch = texts[i : i + batch_size]
                inputs = self.tokenizer(
                    batch, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=seq_len
                ).to(self.device)
                
                self.model(**inputs)
        
        # 清理 Hooks
        for h in self.hooks: h.remove()


    def generate_visualizations(self):
        """分析结果并绘图，同时生成文本报告"""
        logger.info("📊 Generating Heatmaps & Report...")
        
        # 创建文本报告路径
        report_path = os.path.join(self.output_dir, "analysis_summary.txt")
        
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"MoE Analysis Report\nModel: {MODEL_ID}\n{'='*40}\n\n")
            
            for layer_idx, stats in self.layer_stats.items():
                co_matrix = stats["co_matrix"]
                counts = stats["counts"]
                
                # 计算条件概率矩阵 P(j | i)
                # 避免除以 0
                safe_counts = counts.clone()
                safe_counts[safe_counts == 0] = 1.0
                
                # 广播除法 [E, E] / [E, 1]
                prob_matrix = co_matrix / safe_counts.unsqueeze(1)
                prob_np = prob_matrix.numpy()
                
                # === 处理对角线 (设为 NaN 以便绘图时留白) ===
                np.fill_diagonal(prob_np, np.nan)
                
                # === 绘图 ===
                plt.figure(figsize=(10, 9))
                sns.set_context("notebook")
                
                # 只有当只有一层时，才显示具体数值，否则全层分析时字太小
                annot = True if len(self.layer_stats) == 1 and self.num_experts < 32 else False
                
                sns.heatmap(
                    prob_np,
                    cmap="turbo",      # 高对比度颜色
                    square=True,
                    xticklabels=False, # 专家太多不显示具体编号
                    yticklabels=False,
                    annot=annot,
                    fmt=".2f",
                    cbar_kws={'label': 'P(Expert J | Expert I)'}
                )
                
                mode_str = "Single Layer" if len(self.layer_stats) == 1 else "All Layers Scan"
                plt.title(f"Layer {layer_idx} Co-activation ({mode_str})\nModel: {MODEL_ID}")
                plt.xlabel("Expert J (Co-activated)")
                plt.ylabel("Expert I (Pivot)")
                
                filename = f"heatmap_layer_{layer_idx:02d}.png"
                save_path = os.path.join(self.output_dir, filename)
                plt.savefig(save_path, dpi=150)
                plt.close() # 关闭画布释放内存
                
                # === 打印 Top 关联并写入文件 ===
                self._report_top_pairs(layer_idx, prob_np, counts, f)
            
        logger.info(f"✅ Analysis Complete! Results saved to: {self.output_dir}")
        logger.info(f"📄 Detailed Text Report: {report_path}")

    def _report_top_pairs(self, layer_idx, prob_np, counts, file_handle):
        total_activations = counts.sum().item()
        total_tokens = total_activations / self.top_k if self.top_k > 0 else 1.0
        
        # 按激活次数降序排列
        sorted_indices = torch.argsort(counts, descending=True)
        
        header_active = f"\n--- Layer {layer_idx} Top {NUM_TOP_ACTIVE} Active Experts ---"
        print(header_active)
        file_handle.write(header_active + "\n")
        
        for i in range(min(NUM_TOP_ACTIVE, len(counts))):
            idx = sorted_indices[i].item()
            cnt = counts[idx].item()
            
            # === 修改：计算相对于 Token 总数的概率 ===
            ratio = cnt / total_tokens if total_tokens > 0 else 0
            
            # 格式: Expert XX (Count/TotalTokens) Ratio
            line = f"Expert {idx:02d} ({int(cnt)}/{int(total_tokens)}) {ratio:.2%}"
            print(line)
            file_handle.write("  " + line + "\n")

        """在控制台打印 Top 关联对，并写入文件"""
        # 将矩阵展平并排序，找到概率最高的索引
        flat_indices = np.argsort(np.nan_to_num(prob_np).flatten())[::-1]
        
        header = f"\n--- Layer {layer_idx} Strongest Correlations ---"
        print(header)
        file_handle.write(header + "\n")
        
        # 1. 全局 Top Pairs (概率最高的几对)
        file_handle.write(">>> Global Top Pairs (P(J|I)):\n")
        count_printed = 0
        for idx in flat_indices:
            r = idx // self.num_experts
            c = idx % self.num_experts
            val = prob_np[r, c]
            
            # 过滤：如果是 NaN，则忽略
            if np.isnan(val) or (ENABLE_COUNT_FILTER and counts[r] < MIN_COUNT_THRESHOLD):
                continue
            
            line = f"Exp {r:02d} -> Exp {c:02d} : {val:.1%} (Pivot Count: {int(counts[r])})"
            
            # 控制台打印前 5 个，让你一眼看到最强的
            if count_printed < 5:
                print(line)
            
            # 文件写入前 20 个
            if count_printed < 20:
                file_handle.write("  " + line + "\n")
            
            if count_printed >= 20:
                break

            count_printed += 1
            
        # 2. 每个专家的 Top Co-activators (更详细的列表)
        file_handle.write("\n>>> Top Co-activators per Expert:\n")
        for r in range(self.num_experts):
            if ENABLE_COUNT_FILTER and counts[r] < MIN_COUNT_THRESHOLD: continue

            # 获取该专家的行
            row = np.nan_to_num(prob_np[r])
            # 排序找到 Top NUM_COACTIVATORS
            top_indices = np.argsort(row)[::-1][:NUM_COACTIVATORS]
            
            partners = []
            for c in top_indices:
                val = row[c]
                partners.append(f"Exp{c:02d}({val:.0%})")
            
            file_handle.write(f"  Expert {r:02d}: " + ", ".join(partners) + "\n")
        
        file_handle.write("\n")


# ================= 🚀 主程序入口 =================

if __name__ == "__main__":
    analyzer = MoEContextAnalyzer(MODEL_ID, OUTPUT_DIR)
    
    # 根据配置注册 Hook
    # 如果 TARGET_LAYER 是数字，只 Hook 那一层
    # 如果 TARGET_LAYER 是 None，Hook 所有层
    analyzer.register_hooks(target_layer=TARGET_LAYER)
    
    # 运行
    analyzer.run_inference(NUM_SAMPLES, BATCH_SIZE, MAX_SEQ_LEN)
    analyzer.generate_visualizations()