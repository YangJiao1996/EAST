import icdar


def main():
    data_generator = icdar.generator()
    data = next(data_generator)
    print(data[0])

if __name__ == '__main__':
    main()