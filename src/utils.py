import yaml
import os

CONFIG_PATH = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')

def load_config():
    """加载YAML配置文件"""
    # 确保使用绝对路径
    abs_config_path = os.path.abspath(CONFIG_PATH)
    if not os.path.exists(abs_config_path):
        raise FileNotFoundError(f"配置文件不存在: {abs_config_path}")
    try:
        with open(abs_config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config
    except Exception as e:
        print(f"加载或解析配置文件 {abs_config_path} 时出错: {e}")
        raise

# Version: 1.0
# Updated: 2025-05-12 14:51:59
# Version: 1.1
# Updated: 2025-05-12 16:50:46 