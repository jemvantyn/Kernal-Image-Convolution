from img import image as img
import timeit

def main():
    start_time = timeit.default_timer()
    print("Loading image... ", end='')
    test = img('hamburglar.jfif')
    print("Done in", "%.4f" % (timeit.default_timer() - start_time), "seconds.")
    start_time = timeit.default_timer()
    # print("Converting to greyscale... ", end='')
    # test.greyscale_avg()
    # print("Done in", "%.4f" % (timeit.default_timer() - start_time), "seconds.")
    # start_time = timeit.default_timer()
    print("Convoluting... ", end='')
    test.convolute('negative 3')
    print("Done in", "%.4f" % (timeit.default_timer() - start_time), "seconds.")
    start_time = timeit.default_timer()
    print("Converting to decimal... ", end='')
    test.to_decimal()
    print("Done in", "%.4f" % (timeit.default_timer() - start_time), "seconds.")
    start_time = timeit.default_timer()
    print("Displaying image... ", end='')
    test.disp_img()
    print("Done in", "%.4f" % (timeit.default_timer() - start_time), "seconds.")
    start_time = timeit.default_timer()

if __name__ == '__main__':
    main()
