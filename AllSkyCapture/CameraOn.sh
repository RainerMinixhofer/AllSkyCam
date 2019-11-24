#/bin/bash
#Switches 5V USB Power supply line of USB 3.0 Port of ZWO Camera on
echo "35" > /sys/class/gpio/export
echo "out" > /sys/class/gpio/gpio35/direction
echo "1" > /sys/class/gpio/gpio35/value
