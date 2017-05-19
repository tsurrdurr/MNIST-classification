import fileinput
import struct
from sklearn.externals import joblib
root = "YOUR:\\ROOT\\PATH"
def main():
    labels_path = root + "train-labels-idx1-ubyte.gz"
    labels = get_labels(labels_path)
    joblib.dump(labels, root + "labels.joblib.pkl")

    images_path = root + "train-images-idx3-ubyte.gz"
    images = get_grayscaled_images(images_path)
    joblib.dump(images, root + "grayscaled_images_numeric.joblib.pkl")

def get_labels(labels_path):
    g = fileinput.FileInput(labels_path, openhook=fileinput.hook_compressed)
    x = g.__next__()
    head = []
    for i in range(2):
        head.append(struct.unpack(">I", x[4 * i:4 * i + 4])[0])
    magic, n_labels = head
    print("magic={}\nlabels={}".format(*head))

    labels = []
    j = 8  # byte index on current chunk
    while len(labels) < n_labels:
        try:
            val = x[j]
        except IndexError:
            # read a new chuck from file
            x = g.__next__()
            j = 0
            val = x[j]
        labels.append(val)
        j += 1
    return  labels

def get_grayscaled_images(images_path):
    f = fileinput.FileInput(images_path, openhook=fileinput.hook_compressed)
    x = f.__next__()
    head = []
    for i in range(4):
        head.append(struct.unpack(">I", x[4*i:4*i+4])[0])
    magic, n_images, rows, columns = head
    print("magic={}\nimages={}\nrows={}\ncols={}".format(*head))
    j = 16 # index in current chunk
    images = []
    for i in range(n_images):
        img = [[0] * rows for i in range(columns)]
        for r in range(rows):
            for c in range(columns):
                try:
                    val = x[j]
                except IndexError:
                    # need to read a new chunk of data from file
                    x = f.__next__()
                    j = 0
                    val = x[j]
                if val > 170:
                    img[r][c] = 2
                elif val > 85:
                    img[r][c] = 1
                else:
                    img[r][c] = 0
                j+=1
        images.append(img)
    return images

if  __name__ =='__main__':main()