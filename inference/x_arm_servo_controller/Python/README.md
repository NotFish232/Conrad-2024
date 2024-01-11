# xarm

## Table of Content
* [Description](#description)
* [Software Dependencies](#software-dependencies)
* [License](#license)
* [Installation (Linux, MacOS and Raspberry Pi)](#Installation-(Linux,-MacOS-and-Raspberry-Pi))
* [Installation (Windows)](#installation-(windows))
* [Methods and examples](#methods-and-examples)
    * [*class* Servo](#servo)
    * [*class* Controller](#controller)
    * [setPosition](#setposition)
    * [getPosition](#getposition)
    * [servoOff](#servooff)
    * [getBatteryVoltage](#getbatteryvoltage)
* [Things left To-Do](#to-do)

___
## Description

A [Python module](https://pypi.org/project/xarm/) for controlling the Lewan-Soul, Lobot, HiWonder xArm and LeArm. Works on Linux, MacOS and Windows.

## Software Dependencies

* [Python](http://python.org/)
* [PIP](https://pypi.org/project/pip/)
* [cython-hidapi](https://github.com/trezor/cython-hidapi)

## License

[MIT Open Source Initiative](https://opensource.org/licenses/MIT)

## Installation (Linux, MacOS and Raspberry Pi)

    $ sudo apt-get install python-dev libusb-1.0-0-dev libudev-dev
    $ sudo pip install --upgrade setuptools
    $ sudo pip install hidapi
    $ sudo pip install xarm

To enable the xArm/LeArm USB interface it is necessary to add udev rules.

* Debian Linux:

        $ sudo nano /usr/lib/udev/rules.d/99-xarm.rules

* Raspbian (RPi) Linux:

        $ sudo nano /etc/udev/rules.d/99-xarm.rules

Copy this line into the file and then save and exit:

    SUBSYSTEM=="usb", ATTR{idVendor}=="0483", ATTR{idProduct}=="5750", MODE="0660", GROUP="plugdev"

If after adding the udev rules you get an OSError using the open statement, perform one of the following:

1. Reload rules from terminal.

        $ sudo udevadm control --reload-rules && udevadm trigger

1. Restart the comuter.

## Installation (Windows)

    > pip install --upgrade setuptools
    > pip install hidapi
    > pip install xarm

## Methods and examples

### Connecting to the xArm

This example is the bare minimum to connect to the xArm via USB and read the battery voltage.

```py
import xarm

# arm is the first xArm detected which is connected to USB
arm = xarm.Controller('USB')
print('Battery voltage in volts:', arm.getBatteryVoltage())
```
Output:
> Battery voltage: 7.677

The Controller can connect to a specific xArm by appending the serial number to "USB". To find the serial number of your xArm, turn on debug, then after connecting one xArm, run this code.

```py
import xarm

arm = xarm.Controller('USB', debug=True)
```
Output:
>Serial number: 497223563535

Now knowing the serial number you can connect to a specific xArm like this example using two.

```py
import xarm

arm1 = xarm.Controller('USB497223563535')
arm2 = xarm.Controller('USB497253743535')

print('Battery voltage of arm1 in volts:', arm1.getBatteryVoltage())

print('Battery voltage of arm2 in volts:', arm2.getBatteryVoltage())
```
Output:

>Battery voltage of arm1 in volts: 7.677\
>Battery voltage of arm2 in volts: 7.662

In the same way, the Controller can connect to one or more serailly connected xArms.

### Classes and Methods

<a id="servo"></a>
*class* **Servo**(*servo_id*__[__, *position=500*__]__)
<dl><dd>
Returns a <i>Servo</i> object. The Servo class is used to hold the position of a given servo in units and angle.

Properties include:
<ul>
<li>servo_id - ID of servo.</li>
<li>position - Position of servo in units (0 to 1000).</li>
<li>angle - Angle of servo in degrees (-125.0 to 125.0 in 0.25 degree integral).</li>
</ul>

The <em>position</em> paramter may be specified with an <em>int</em> value between 0 and 1000. When set to a <em>float</em> value between -125.0 to 125.0, the <em>angle</em> value is set in degrees and rounded to a 0.25 degree integral. There are 4 servo units per degree of angle.

>Note: In Python, an <em>int</em> value does not have a decimal point (500). A <em>float</em> value has a decimal point (0.0).

Setting position will also set the corresponding angle and visa-versa. If <em>position</em> is not specified, it will default to '500' and <em>angle</em> will be set to '0.0'.

```py
import xarm

arm = xarm.Controller('USB')

# define servo as servo ID 1 with position 300
servo = xarm.Servo(1, 300)

print('servo ID:', servo.servo_id)
print('servo position:', servo.position)
print('servo angle:', servo.angle)
```
Output:
>servo ID: 1\
>servo position: 300\
>servo angle: -50.0
</dd></dl>

<a id="controller"></a>
*class* **Controller**(*com_port*__[__, *debug=False*__]__)
<dl><dd>
Returns a <i>Controller</i> object. The Controller class connects Python to the xArm. The port to connect to the xArm through is determined by <i>com_port</i> which can be a serial port (<code>COM5</code>) or USB port (<code>USB</code>). Multiple xArms may be connected. If more than one xArm is attached by USB, each can be identified by appending the serial number to 'USB' (<code>USB497223563535</code>). 

Optionally, when <i>debug</i> is <code>True</code>, communication diganostic information will be printed to the terminal.

```py
# attach to xArm connected to USB
arm1 = xarm.Controller('USB')
# attach to USB connected xArm with serial number '497223563535'
arm2 = xarm.Controller('USB497223563535')
# attach to xArm connected to serial port 'COM5'
arm3 = xarm.Controller('COM5')
# enable debug
arm4 = xarm.Controller('COM6', True)        # positional argument
arm5 = xarm.Controller('COM7', debug=True)  # named argument
```
</dd></dl>

<a id="setposition"></a>
**setPosition**(*servos*__[__, *position=None*, *duration=1000*, *wait=False*__]__)
<dl><dd>
Moves one or more <em>servos</em> to a specified <em>position</em> over a <em>duration</em> and optionallly <em>wait</em>s during the duration.

When <em>servos</em> is an <em>int</em> value, it represents a servo ID and the <em>position</em> parameter is required.

The <em>position</em> paramter may be an <em>int</em> to indicate a unit position (0 to 1000) or a <em>float</em> to indicate an angle in degrees (-125.0 to 125.0).

```py
# Set servo ID 1 to position 500.
setPosition(1, 500)
```

When <em>servos</em> is a <em>Servo</em> object, the <em>position</em> parameter is ignored.The Servo parameter holds the position and angle.

When <em>servos</em> is a <em>list</em>, it may contain <em>Servo</em> objects or servo ID and position pairs and the <em>position</em> parameter is ignored.

```py
import xarm

arm = xarm.Controller('USB')

servo1 = xarm.Servo(1)       # assumes default unit position 500
servo2 = xarm.Servo(2, 300)  # unit position 300
servo3 = xarm.Servo(3, 90.0) # angle 90 degrees

# sets servo 1 to unit position 300 and waits the default 1 second
# before returning
arm.setPosition(1, 300, wait=True)

# sets servo 2 to unit position 700 and moves the servo at a
# rate of 2 seconds
arm.setPosition(2, 700, 2000, True)

# sets servo1 to 45 degrees and waits the default 1 second
# before returning
arm.setPosition(3, 45.0, wait=True) 

# sets servo 2 to position 300 as defined above but continues to
# the next method before completing movement
arm.setPosition(servo2) 

# sets servos 1-3 as defined and continues without waiting
arm.setPosition([servo1, servo3])

# sets servos 1 to unit position 200 and servo 2 to 90 degrees
arm.setPosition([[1, 200], [2, 90.0]], wait=True) 

# Servo object and servo ID/position pairs can be combined
arm.setPosition([servo1, [2, 500], [3, 0.0]], 2000)
```
</dd></dl>

<a id="getposition"></a>
**getPosition**(*servos*__[__, *degrees=False*__]__)
<dl><dd>
Returns the current <em>position</em> of one or more <em>servos</em>.

By default, the <em>unit position</em> is returned. When <em>degrees</em> is <code>True</code>, the <em>angle</em> is returned.

The <em>servos</em> parameter may be a servo ID (1 to 6) or a <em>Servo</em> object or a list of one or more <em>Servo</em> objects.

```py
import xarm

arm = xarm.Controller('USB')

servo1 = xarm.Servo(1)
servo2 = xarm.Servo(2)
servo3 = xarm.Servo(3)

# Gets the position of servo 1 in units
position = arm.getPosition(1)
print('Servo 1 position:', position)

# Gets the position of servo 2 as defined above
position = arm.getPosition(servo2)
print('Servo 2 position:', position)

# Gets the position of servo 3 in degrees
position = arm.getPosition(3, True)
print('Servo 3 position (degrees):', position)

# Gets the position of servo 2 as defined above
# It is not necessary to set the degreees parameter
# because the Servo object performes that conversion
position = arm.getPosition([servo1, servo2, servo3])
print('Servo 1 position (degrees):', servo1.angle)
print('Servo 2 position (degrees):', servo2.angle)
print('Servo 3 position (degrees):', servo3.angle)
```
</dd></dl>

<a id="servooff"></a>
**servoOff**(__[__*servos=None*__]__)
<dl><dd>
Turns off motor of one or more servos. If <em>servos</em> paramter is not specified, will turn off all servo motors.

```py
import xarm

arm = xarm.Controller('USB')

servo2 = xarm.Servo(2)
servo5 = xarm.Servo(5)
servo6 = xarm.Servo(6)

# Turns off servo motor 1
arm.servoOff(1)

# Turns off servo motor 1
arm.servoOff(servo2)

# Turns off servo motors 3 and 4
arm.servoOff([3, 4])

# Turns off servo motors 5 and 6
arm.servoOff([servo5, servo6])
```
</dd></dl>

<a id="getbatteryvoltage"></a>
**getBatteryVoltage**()
<dl><dd>
Returns battery or power supply voltage on volts.

```py
import xarm

arm = xarm.Controller('USB')

battery_voltage = arm.getBatteryVoltage()
print('Battery voltage (volts):', battery_voltage)
```
</dd></dl>
