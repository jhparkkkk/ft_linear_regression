import sys as sys


theta0= 2.4563684419831597e-16
theta1 = -0.8561394201864889
def estimate(x):
    print('x:', x)
    x = (x - 101066.25) / 51565.1899106445
    print('x:', x)
    print(theta0 + (theta1 * x))
    return theta0 + (theta1 * x)

def getUserInput():
    try:
        assert len(sys.argv) <= 2, "Only one arg allowed"
        if len(sys.argv) == 1:
            while (True):
                try:
                    userInput = input("What is the mileage?\n")
                    return int(userInput)
                except EOFError:
                    return userInput
                    exit()
        else:
            userInput = sys.argv[1]
        return userInput
    except AssertionError as e:
        print("AssertionError:", e)

if __name__ == "__main__":
    userInput=getUserInput()
    if userInput:
        price_scaled = estimate(userInput)
        print(price_scaled)
        price = 0
        price = price_scaled * 1291.8688873961717
        price = price + 6331.833333333333
        print(price)

