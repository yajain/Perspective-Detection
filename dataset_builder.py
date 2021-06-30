import os

years = os.listdir('dataset/')

labels = []
for year in years:
    for month in os.listdir('dataset/'+year):
        for file in os.listdir('dataset/'+year+'/'+month):
            if ((file.split('.')[-1]) == 'cfx' or (file.split('.')[-1] == 'CFX')):
                f = open('dataset/'+year+'/'+month+'/'+file, 'r')
                flag = 0
                for line in f:
                    if not(line.find('PERSP(DEF=') == -1):
                        l = line[line.find('=')+1:-2]
                        l = l[1:-1]
                        l = l.replace('(','')
                        l = l.replace(')','')
                        a = l.split(',')
                        a.append('dataset/'+year+'/'+month+'/'+file.split('.')[0]+'_Original')
                        labels.append(a)
                        flag = 1
                    if flag == 1:
                        break


for label in labels:
    if not(len(label) == 9):
        label.remove('\t')
        print(label)

f = open('labels.txt','w')
for a in labels:
    f.write(str(a))
    f.write('\n')
f.close()

years = os.listdir('dataset/')

img_path = []

for year in years:
    for month in os.listdir('dataset/'+year):
        for file in os.listdir('dataset/'+year+'/'+month):
            if os.path.isdir('dataset/'+year+'/'+month+'/'+file):
                for image in os.listdir('dataset/'+year+'/'+month+'/'+file):
                    if ((image.split('.')[0].split('_')[-1] == 'Original') or (image.split('.')[0].split('_')[-1] == 'Original')):
                        img_path.append('datatset/'+year+'/'+month+'/'+file+'/'+image)
