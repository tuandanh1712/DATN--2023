import RPi.GPIO as GPIO
import time


def unlock1():
    '''
    pin: pin which you attach on your pi with solenoid lock, default is 26
    '''
    #print(pin)
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(31, GPIO.OUT)
    GPIO.output(31,GPIO.HIGH)
def lock1():
    '''
    pin: pin which you attach on your pi with solenoid lock, default is 26
    '''
    #GPIO.setmode(GPIO.BCM)
    GPIO.setup(31,GPIO.OUT)
    GPIO.output(31,GPIO.LOW)
    

if __name__ == '__main__':
    unlock1()
    time.sleep(2)
    lock1()
    GPIO.cleanup(7)

