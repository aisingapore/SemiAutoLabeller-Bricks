import matplotlib.pyplot as plt

def barplot(x, y, figsize, title, x_label, y_label):
    f, ax = plt.subplots(figsize=figsize)
    plt.bar(x, y)
    ax.set_title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.show()