import os

years = os.listdir('dataset/')

img_paths = []

for year in years:
    for month in os.listdir('dataset/'+year):
        for file in os.listdir('dataset/'+year+'/'+month):
            if os.path.isdir('dataset/'+year+'/'+month+'/'+file):
                for image in os.listdir('dataset/'+year+'/'+month+'/'+file):
                    if ((image.split('.')[0].split('_')[-1] == 'Original') or (image.split('.')[0].split('_')[-1] == 'Original')):
                        img_paths.append('dataset/'+year+'/'+month+'/'+file+'/'+image)

abels = []

for image in img_paths:
    label_file = image.split('/')
    png = label_file[-1]
    cfx = png.split('.')[0].replace('_Original', '.cfx')
    label_file = label_file[0:-2]
    label_file = '/'.join(label_file)
    label_file = label_file+'/'+cfx
    if os.path.exists(label_file):
        labels.append(label_file)
    else:
        labels.append('No file')

targets = []

for label in labels:
    if os.path.exists(label):
        f = open(label, 'r')
        flag = 0
        for line in f:
            if not(line.find('PERSP(DEF=') == -1):
                l = line[line.find('=')+1:-2]
                l = l[1:-1]
                l = l.replace('(','')
                l = l.replace(')','')
                a = l.split(',')
                targets.append([a, label])
                flag = 1
            if flag == 1:
                break
    else:
        targets.append('No file')

count = 0
for label in targets:
    if (label == 'No file'):
        count = count + 1

for i in range(len(labels)):
    if labels[i] == 'No file':
        img_paths[i] = 'No file'

for i in range(418):
    img_paths.remove('No file')
    labels.remove('No file')
    targets.remove('No file')
