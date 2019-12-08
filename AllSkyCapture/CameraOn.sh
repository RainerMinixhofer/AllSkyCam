#/bin/bash
#Switches 5V USB Power supply line of USB 3.0 Port of ZWO Camera on
echo "35" > /sys/class/gpio/export
#give kernel time to generate the required control files
sleep 0.1
echo "out" > /sys/class/gpio/gpio35/direction
echo "1" > /sys/class/gpio/gpio35/value
