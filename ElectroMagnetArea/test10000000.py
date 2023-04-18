import time
start = time.time()
for i in range(100000000):
    a = i**2
endtime = time.time()

print("耗時",endtime-start)