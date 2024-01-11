import hid
import time


def pad_or_truncate(some_list, target_len):
    return some_list[:target_len] + [0]*(target_len - len(some_list))

dev = hid.device()
dev.open(0x0483, 0x5750)

print("Manufacturer: %s" % dev.get_manufacturer_string())
print("Product: %s" % dev.get_product_string())
print("Serial No: %s" % dev.get_serial_number_string())

# enable non-blocking mode
dev.set_nonblocking(1)

# write some data to the device
print("Write the data")
data = [0, 0x55, 0x55, 2, 0x0f]
data = pad_or_truncate(data, 65)
dev.write(data)

# wait
#time.sleep(0.05)

# read back the answer
print("Read the data")
response = []
while True:
    d = dev.read(64, 50)    
    if d:
        response.extend(d)
    else:
        break

print(f"{(response[5] * 256 + response[4])/1000} VDC")

print("Closing the device")
dev.close()