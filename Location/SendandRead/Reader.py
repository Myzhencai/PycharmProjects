# serial_rxd.py
import time
import serial

# set serial port initialized parameters
com = serial.Serial(
    port = '/dev/ttyUSB0',
    baudrate = 115200,
    bytesize = serial.EIGHTBITS,
    parity = serial.PARITY_NONE,
    stopbits = serial.STOPBITS_ONE,
)

# wait 1s for serial port initialization
time.sleep(1)

# received data and print in hex string form
while 1:
    rxd_num = com.inWaiting()
    if rxd_num > 0:
        rxd = com.read(rxd_num)
        print(str(rxd.hex()))
