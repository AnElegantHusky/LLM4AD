import colorama
from colorama import Fore, Style

# 关键：在这里进行一次性初始化
# autoreset=True 是一个非常有用的设置，它会在每次 print 后自动重置颜色
colorama.init(autoreset=True)

# 定义几个辅助函数
def print_error(message):
    """打印一条红色的、加粗的错误信息"""
    print(f"{Fore.RED}{Style.BRIGHT}Error： {message}")

def print_warning(message):
    """打印一条黄色的警告信息"""
    print(f"{Fore.YELLOW}Warning： {message}")

def print_success(message):
    """打印一条绿色的成功信息"""
    print(f"{Fore.GREEN}Success： {message}")

def print_info(message):
    """打印一条蓝色的检查信息"""
    print(f"{Fore.BLUE}Info： {message}")

# 你也可以直接导出颜色，以防其他地方需要自定义
# (但通常更推荐使用上面的函数)
c_error = Fore.RED + Style.BRIGHT
c_warning = Fore.YELLOW
c_success = Fore.GREEN
c_info = Fore.BLUE
c_reset = Style.RESET_ALL