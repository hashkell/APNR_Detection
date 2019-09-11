import glob

text_files = glob.glob('./data/*.txt')

for file in text_files:
    with open(file, 'r') as f:
        with open(file, 'w') as F:
            for annotation in f:
                annotation = annotation.rstrip()
                annotation = annotation.split(' ')
                annotation = [eval(x) for x in annotation]
                F.write(str(object=annotation))
