# modifed from https://github.com/salesforce/awd-lstm-lm/blob/master/data/enwik8/prep_enwik8.py

import os
import sys

data_dir = "data"
if os.path.exists(os.path.join(data_dir, "train.txt")):
    print("Tokenized enwik8 already exists - skipping processing")
    sys.exit()

data = open(os.path.join(data_dir, "enwik8"), "rb").read()

print("Length of enwik8: {}".format(len(data)))

num_test_chars = 5000000

train_data = data[: -2 * num_test_chars]
valid_data = data[-2 * num_test_chars : -num_test_chars]
test_data = data[-num_test_chars:]

for fn, part in [
    ("train.txt", train_data),
    ("valid.txt", valid_data),
    ("test.txt", test_data),
]:
    fn = os.path.join(data_dir, fn)
    print("{} will have {} bytes".format(fn, len(part)))
    print("- Tokenizing...")
    part_str = " ".join([str(c) if c != ord("\n") else "\n" for c in part])
    print("- Writing...")
    f = open(fn, "w").write(part_str)
    f = open(fn + ".raw", "wb").write(part)
