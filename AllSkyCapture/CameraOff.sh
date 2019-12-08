#/bin/bash
#Switches 5V USB Power supply line of USB 3.0 Port of ZWO Camera off
echo "0" > /sys/class/gpio/gpio35/value
echo "35" > /sys/class/gpio/unexport
