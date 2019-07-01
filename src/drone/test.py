from time import sleep
import tellopy


def handler(event, sender, data, **args):
    drone = sender
    if event is drone.EVENT_FLIGHT_DATA:
        print(data)


def test():
    drone = tellopy.Tello()
    try:
        drone.subscribe(drone.EVENT_FLIGHT_DATA, handler)

        drone.connect()
        drone.wait_for_connection(6000.0)
        
        sleep(5)
        drone.takeoff()
        sleep(5)
        drone.left(10)
        sleep(5)
        drone.right(10)
        sleep(5)
        # drone.forward(10)
        # sleep(2)
        # drone.backward(10)
        # sleep(2)
        drone.down(50)
        sleep(5)
        drone.land()
        sleep(5)
    except Exception as ex:
        print(ex)
        drone.land()
        sleep(5)
    finally:
        drone.quit()

if __name__ == '__main__':
    test()
