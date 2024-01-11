README.md
# hidapi
## Installation
    $ sudo apt install openssh-server openssh-client python-dev libusb-1.0-0-dev libudev-dev python3 python3-venv python3-pip
    python3 -m venv env
    source env/bin/activate
    $ sudo pip3 install --upgrade setuptools
    $ pip3 install hidapi

## API Docs
https://trezor.github.io/cython-hidapi/api.html

ms-python.python
    Press button: "Install in 'SSH: mac-laptop"



## Troubleshooting Mac
No module named -venv
    pip3 install virtualenv

ModuleNotFoundError
    pip3 install hidapi

OSError at dev.open(0x0483, 0x5750)
    

## Fix "Access Denied"
USBError: [Errno 13] Access denied (insufficient permissions)

1. Edit /etc/udev/rules.d/99-com.rules (RPi), /usr/lib/udev/rules.d (Ubuntu). 99-com.rules may be named differently on your instance. You can make your own, e.g. 99-my-rules.rules
    
       sudo nano /etc/udev/rules.d/99-com.rules
    
1. Add line:

       SUBSYSTEM=="usb", ENV{DEVTYPE}=="usb_device", ATTRS{idVendor}=="0483", ATTRS{idProduct}=="5750", GROUP="plugdev", MODE="0660"

1. Run commands:

       sudo udevadm control --reload
       sudo udevadm trigger

1. Restart the RPi

       sudo shutdown -r now

## Fix "Resource busy"
usb.core.USBError: [Errno 16] Resource busy

Add after setting _dev_:

    # To resolve possible "Resource busy" error
    if dev.is_kernel_driver_active(0):
        dev.detach_kernel_driver(0)