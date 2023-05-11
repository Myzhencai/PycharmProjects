import os, re
import webbrowser


class WeNetWork_open_browser():
    def execCmd(self, cmd):  # 返回CMD命令输出的内容
        r = os.popen(cmd)
        text = r.read()
        r.close()
        return text

    def get_IP(self):
        print("hello")
        # # print(self.execCmd("ipconfig")) # 调用cmd命令
        # result = self.execCmd("ifconfig")
        # resultlist = result.split("\n")
        # for strings in resultlist:
        #     print("hello")
        #     print(strings)
        #     if "wlo1"
        # pat2 = "无线局域网适配器 WLAN:?\n.*\n.*\n.*\n.*IPv4 地址 [\. ]+:(.*)"
        # pat2 = "inet:"
        # print(result)
        # pat2 = "wlo1："
        # IP = re.findall(pat2, result)[0]
        # IP = ''.join(IP.split())  # 去掉空格
        #
        # url = r"http://" + str(IP) + ":" + "8001"
        # return url

    def open_browser(self):
        webbrowser.open(self.get_IP())


if __name__ == '__main__':
    open_browser_start = WeNetWork_open_browser()
    # open_browser_start.open_browser()
    open_browser_start.get_IP()