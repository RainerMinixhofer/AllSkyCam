# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 07:28:18 2019

@author: Rainer Minixhofer
"""

import smbus

class IRSensor :
  # RAM Registers
  __RAW1    = 0x04
  __RAW2    = 0x05
  __TA      = 0x06
  __TO1     = 0x07
  __TO2     = 0x08
  # EEPROM Registers (address of EEPROM + 2x20 see Section 8.4.5 of MLX90614 Datasheet)
  __TOMAX   = 0x00 + 0x20
  __TOMIN   = 0x01 + 0x20
  __PWMCTRL = 0x02 + 0x20
  __TARANGE = 0x03 + 0x20
  __EMISSIV = 0x04 + 0x20
  __CONFIG1 = 0x05 + 0x20
  __KE      = 0x0F + 0x20
  __CONFIG2 = 0x19 + 0x20

  @staticmethod
  def getI2CBusNumber():
    # Gets the I2C bus number /dev/i2c-#
    return 2

  @classmethod

  def __init__(self, address=0x5a, debug=True):
    self.address = address
    self.debug = debug
    self.bus = smbus.SMBus(self.getI2CBusNumber())
    if debug:
        print("I2C Bus Number: %d" % self.getI2CBusNumber())

  def errMsg(self, error):
    print("Error accessing 0x%02X(%X): Check your I2C address" % (self.address, error))
    return -1

  # Only write16, readU16 and readS16 commands
  # are supported. (see MLX90614 family, section 8.4.2)
  def readU16(self, reg, little_endian=True):
    "Reads an unsigned 16-bit value from the I2C device"
    try:
      result = 0xFFFA
      while result == 0xFFFA:
          result = self.bus.read_word_data(self.address, reg) & 0xFFFF
      # Swap bytes if using big endian because read_word_data assumes little
      # endian on ARM (little endian) systems.
      if not little_endian:
        result = ((result << 8) & 0xFF00) + (result >> 8)
      if (self.debug):
        print("I2C: Device 0x%02X returned 0x%04X from reg 0x%02X" % (self.address, result & 0xFFFF, reg))
      return result
    except IOError as err:
      return self.errMsg(err)

  def readS16(self, reg, little_endian=True):
    "Reads a signed 16-bit value from the I2C device"
    try:
      result = self.readU16(reg, little_endian)
      if result > 32767: result -= 65536
      return result
    except IOError as err:
      return self.errMsg(err)

  def write16(self, reg, value):
    "Writes a 16-bit value to the specified register/address pair"
    try:
      self.bus.write_word_data(self.address, reg, value)
      if self.debug:
        print(("I2C: Wrote 0x%02X to register pair 0x%02X,0x%02X" %
         (value, reg, reg+1)))
    except IOError as err:
      return self.errMsg(err)

  def Ta(self, imperial=False):
    # Temperatur Factor set to0.02 degrees per LSB
    # (measurement resolution of the MLX90614)
    tempFactor = 0.02
    tempData = self.readU16(self.__TA)
    tempData = (tempData * tempFactor)-0.01
    if imperial:
        #Ambient Temperature in Farenheit
        return ((tempData - 273.15)*1.8) + 32
    else:
        #Ambient Temperature in Celsius
        return tempData - 273.15
    
  def Tobj(self, imperial=False):
    # Temperatur Factor set to 0.02 degrees per LSB
    # (measurement resolution of the MLX90614)
    tempFactor = 0.02
    tempData = self.readU16(self.__TO1)
    tempData = (tempData * tempFactor)-0.01
    if imperial:
        #Ambient Temperature in Farenheit
        return ((tempData - 273.15)*1.8) + 32
    else:
        #Ambient Temperature in Celsius
        return tempData - 273.15
    
sensor = IRSensor(debug=False)
print("Ambient Temperature: %5.2f °C" % sensor.Ta())
print("Object Temperature: %5.2f °C" % sensor.Tobj())
