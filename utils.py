from datetime import datetime
import pytz

def get_current_time(timezone='Asia/Shanghai') -> str:
    # 获取本地时区
    local_timezone = pytz.timezone(timezone)

    # 获取当前时间
    now = datetime.now(local_timezone)

    # 格式化为包含年月日时分的字符串
    formatted_time = now.strftime("%Y%m%d%H%M")

    return formatted_time