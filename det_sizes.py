def correct_size(x):
    if (x - 4) % 2 == 0:
        x = x - 4
        if (x - 8) % 2 == 0:
            x = (x - 8) / 2
            if (x - 8) % 2 == 0:
                x = (x - 8) / 2
                if (x - 8) % 2 == 0:
                    x = (x - 8) / 2
                    if (x - 8) % 2 == 0:
                        x = (x - 8) / 2
                        if x % 2 == 0:
                            return True
    return False


for i in range(284, 4098, 2):
    if correct_size(i):
        print(i)
