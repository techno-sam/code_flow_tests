import random


def test(a: int):
    if a == 42:
        print("Life, the Universe, and Everything")
    elif a == 24:
        print("gnihtyrevE dna ,esrevinU eht ,efiL")
        return
    else:
        print("Nope")

    print("EOB: Done")


def test2(a: int):
    if a == 42:
        print("Life, the Universe, and Everything")
    else:
        if a == 24:
            print("gnihtyrevE dna ,esrevinU eht ,efiL")
            return
        else:
            if a == 4224:
                print("Life, the Universe, and Everything|gnihtyrevE dna ,esrevinU eht ,efiL")
            else:
                print("Nope")
            print("Semi-inner")

    print("EOB: Done")

def basic(a: int):
    if a == 12:
        print("12")
        return
        # noinspection PyUnreachableCode
        print("Dead code")
    else:
        print("Hello")
    print("EOB: Done")
    print("Done2")


def basic2(a: int):
    if a == 12:
        print("12")
        print("Undead code")
    else:
        print("Hello")
    print("EOB: Done")
    print("Done2")

def basic3(a: int):
    if a == 12:
        print("12")
        print("Undead code")
    else:
        print("Hello")

def basic4(a: int):
    if a == 12:
        print("12")
        print("Undead code")
    print("EOB: Done")

def loop_test():
    for i in range(3):
        print("In-loop", i)
        print("Another in-loop")
    print("Done")

def loop_test2():
    for i in range(3):
        print("In-loop", i)
        if i == 1:
            print("Hello, i==1")
        print("Another in-loop")
    print("Done")

def loop_test3():
    for i in range(4):
        print("In-loop", i)
        if i == 0:
            print("Continuing")
            continue
            print("Dead code")
        if i == 1:
            print("Hello, i==1")
        if i == 2:
            print("Breaking")
            break
            print("Dead code 2")
        else:
            print("Not breaking")
        print("Another in-loop")
    print("Done")

def loop_test3b():
    for i in range(4):
        print("In-loop", i)
        if i == 0:
            print("Continuing")
            continue
            print("Dead code")
        if i == 1:
            print("Hello, i==1")
        else:
            continue
        if i == 2:
            print("Breaking")
            break
            print("Dead code 2")
        else:
            print("Not breaking")
        print("Another in-loop")
    print("Done")

def loop_test4():
    for i in range(3):
        print("In-loop", i)
        if i == 1:
            print("Hello, i==1")
        print("Another in-loop")
    while random.randrange(0, 5) != 3:
        print("Still not 3...")
    print("Done")

def loop_test5():
    for i in range(3):
        print("In-loop", i)
        if i == 1:
            print("Breaking")
            break
        if i == 2:
            pass
        print("Another in-loop")
    else:
        print("Did not break")

    print("Done")

def nested_if_test(a: int, b: int):
    for i in range(3):
        print("i is:", i)
        if a == 4:
            print("a is 4")
            if b == 2:
                print("b is 2")

            print("Done with b check")

    print("Done!")

def nested_if_test2(a: int, b: int):
    for i in range(3):
        print("i is:", i)
        if a == 4:
            print("a is 4")
            if b == 2:
                print("b is 2")

def nested_loop_test():
    for i in range(3):
        print("i is:", i)
        for j in range(3):
            print("j is:", j)

        print("End of j loop")

    print("End of i loop")
    print("Done!")

def nested_loop_test2():
    for i in range(3):
        print("i is:", i)
        for j in range(3):
            print("j is:", j)

    print("End of i loop")
    print("Done!")

def nested_loop_test3():
    for i in range(3):
        print("i is:", i)
        for j in range(3):
            print("j is:", j)

def nested_everything_test(a: int):
    for i in range(3):
        print("i is:", i)
        if i == 1:
            print("i is 1")
            if a == 42:
                print("a is 42")

            print("end of a check")

            for j in range(3):
                print("j is:", j)
            print("end of j loop")
        print("end of i check")

def nested_everything_test2(a: int):
    for i in range(3):
        print("i is:", i)
        if i == 1:
            print("i is 1")
            if a == 42:
                print("a is 42")

            for j in range(3):
                print("j is:", j)

def nested_everything_test3(a: int):
    for i in range(3):
        print("i is:", i)
        if i == 1:
            print("i is 1")
            if a == 42:
                print("a is 42")
                if int("0") == 1:
                    print("This cannot happen")
            else:
                print("a is not 24")

            for j in range(3):
                print("j is:", j)

def nested_everything_test4(a: int, b: int):
    if a == 42:
        if b == 1:
            print("b is 1")
    else:
        print("a is not 42")

    for j in range(3):
        print("j is:", j)

def test_find_primes_in_range(start, end):
    primes = []
    for num in range(start, end + 1):
        if num > 1:  # Check if the number is greater than 1
            is_prime = True
            for i in range(2, num):
                if num % i == 0:
                    is_prime = False
                    break
            if is_prime:
                primes.append(num)
    return primes

def recursive_test(n):
    if n > 0:
        return recursive_test(n - 1) * n
    return 0

def main():
    test(42)
    test(24)

    for i in range(21, 44):
        test(i)

    test(12)