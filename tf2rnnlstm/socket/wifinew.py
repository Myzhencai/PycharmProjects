import socket
import fcntl
import struct
# def get_ip_address(ifname):
#     s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#     return socket.inet_ntoa(fcntl.ioctl(s.fileno(), 0x8915, struct.pack('256s', ifname[:15]))[20:24])
# #get_ip_address('lo')环回地址
# get_ip_address('eth0')#主机ip地址


# '''
# 遇到问题没人解答？小编创建了一个Python学习交流QQ群：531509025
# 寻找有志同道合的小伙伴，互帮互助,群里还有不错的视频学习教程和PDF电子书！
# '''
# def get_local_ip(ifname):
#     import socket, fcntl, struct
#     s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#     inet = fcntl.ioctl(s.fileno(), 0x8915, struct.pack('256s', ifname[:15]))
#     ret = socket.inet_ntoa(inet[20:24])
#     return ret
# print(get_local_ip("eth0"))


import psutil
info = psutil.net_if_addrs()

wlan=info['wlo1']
ipaddress = str(wlan[0]).split(",")
ipnum = ipaddress[1].split('=')[1]
ipnumlist = ipnum[1:-2]
# ipnumlist.pop(0)
# ipnumlist.pop(-1)
# print(str(ipnumlist))
print(ipnumlist)
