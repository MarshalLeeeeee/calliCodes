import random
from os.path import exists, join
from os import listdir, walk
from PIL import Image
import numpy as np
import progressbar
import sys

OPS =  [
   ('+', lambda a, b: a+b),
   ('-', lambda a, b: a-b),
   ('*', lambda a, b: a*b),
   ('x', lambda a, b: a*b),
   ('/', lambda a, b: a//b),
]


def parse_math(s):
   for operator, f in OPS:
       try:
           idx = s.index(operator)
           return f(parse_math(s[:idx]), parse_math(s[idx+1:]))
       except ValueError:
           pass
   return int(s)

def next_unused_name(name):
    save_name = name
    name_iteration = 0
    while exists(save_name):
        save_name = name + "-" + str(name_iteration)
        name_iteration += 1
    return save_name


def add_boolean_cli_arg(parser, name, default=False, help=None):
    parser.add_argument(
        "--%s" % (name,),
        action="store_true",
        default=default,
        help=help
    )
    parser.add_argument(
        "--no%s" % (name,),
        action="store_false",
        dest=name
    )


def create_progress_bar(message):
    widgets = [
        message,
        progressbar.Counter(),
        ' ',
        progressbar.Percentage(),
        ' ',
        progressbar.Bar(),
        progressbar.AdaptiveETA()
    ]
    pbar = progressbar.ProgressBar(widgets=widgets)
    return pbar


def find_files_with_extension(path, extensions):
    for basepath, directories, fnames in walk(path):
        for fname in fnames:
            name = fname.lower()
            if any(name.endswith(ext) for ext in extensions):
                yield join(basepath, fname)

def make_one_hot(indices, size):
    as_one_hot = np.zeros((indices.shape[0], size))
    as_one_hot[np.arange(0, indices.shape[0]), indices] = 1.0
    return as_one_hot

def load_calligraphy(path, max_label, styles,
                     desired_height=None,
                     desired_width=None,
                     value_range=None,
                     force_grayscale=False):
    root_path = path
    X_train, Y_train, X_test, Y_test = None, None, None, None
    idx = 0
    for i in range(len(styles)):
        path = join(root_path,styles[i])
        image_paths = list(find_files_with_extension(path, [".png", ".jpg", ".jpeg"]))
        pb = create_progress_bar("Loading dataset ")
        for fname in pb(image_paths):
            label = int(fname.split('/')[-1].split('-')[0])
            #print(label, max_label)
            if label >= max_label: continue
            image = Image.open(fname)
            width, height = image.size
            if desired_height is not None and desired_width is not None:
                if width != desired_width or height != desired_height:
                    image = image.resize((desired_width, desired_height), Image.BILINEAR)
            else:
                desired_height = height
                desired_width = width

            if force_grayscale: image = image.convert("L")
            img = np.array(image)
            if len(img.shape) == 2: img = img[:, :, None]

            if not idx % 10:
                if X_test is None: X_test = np.array([img], dtype=np.float32)
                else: X_test = np.concatenate((X_test,np.array([img], dtype=np.float32)),axis=0)

                if Y_test is None: Y_test = np.array([label], dtype=np.int32)
                else: Y_test = np.concatenate((Y_test,np.array([label], dtype=np.int32)),axis=0)

            else:
                if X_train is None: X_train = np.array([img], dtype=np.float32)
                else: X_train = np.concatenate((X_train,np.array([img], dtype=np.float32)),axis=0)

                if Y_train is None: Y_train = np.array([label], dtype=np.int32)
                else: Y_train = np.concatenate((Y_train,np.array([label], dtype=np.int32)),axis=0)
            idx += 1

    if value_range is not None:
        X_train = (value_range[0] + (X_train / 255.0) * (value_range[1] - value_range[0]))
        X_test = (value_range[0] + (X_test / 255.0) * (value_range[1] - value_range[0]))
    print("dataset loaded.")
    sys.stdout.flush()
    return X_train, make_one_hot(Y_train,max_label), X_test, make_one_hot(Y_test,max_label)