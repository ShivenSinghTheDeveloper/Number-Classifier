from sklearn.datasets import load_digits
def load_data():
    digits = load_digits()
    x = digits.data
    y = digits.target
    images = digits.images
    return x,y,images
