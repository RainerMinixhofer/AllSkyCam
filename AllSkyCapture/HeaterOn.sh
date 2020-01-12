#/bin/bash
#Switches 5V USB Power supply line of USB 3.0 Port of ZWO Camera on
echo "33" > /sys/class/gpio/export
#give kernel time to generate the required control files
#without this sleep time we get device busy errors
sleep 0.1
echo "out" > /sys/class/gpio/gpio33/direction
echo "1" > /sys/class/gpio/gpio33/value
