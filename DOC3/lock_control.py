import RPi.GPIO as GPIO
import time


def unlock():
    '''
    pin: pin which you attach on your pi with solenoid lock, default is 26
    '''
    #print(pin)
    GPIO.setwarnings(False)
    GPIO.setmode(GPIO.BOARD)
    GPIO.setup(26, GPIO.OUT)
    GPIO.output(26,GPIO.HIGH)
def lock():
    '''
    pin: pin which you attach on your pi with solenoid lock, default is 26
    '''
    #GPIO.setmode(GPIO.BCM)
    GPIO.setup(26,GPIO.OUT)
    GPIO.output(26,GPIO.LOW)
    

if __name__ == '__main__':
    unlock()
    time.sleep(2)
    lock()
    GPIO.cleanup(7)

