
# serial_txd.py
import time
import serial


portpath = '/dev/ttyS0'

com = serial.Serial(
    port = portpath,
    baudrate = 115200,
    bytesize = serial.EIGHTBITS,
    parity = serial.PARITY_NONE,
    stopbits = serial.STOPBITS_ONE,
)

# send hex data 'AA55' per 1s
while 1:
    hex_str = 'AA55'
    com.write(bytes.fromhex(hex_str))
    time.sleep(1)
    rxd_num = com.inWaiting()
    if rxd_num > 0:
        rxd = com.read(rxd_num)
        print(str(rxd.hex()))
    else:
        print("error")

