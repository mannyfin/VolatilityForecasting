from pptx import Presentation # pip install python-pptx
from pptx.util import Inches
from os import listdir
from os.path import isfile, join

filename = "GBPUSD"
# modify the path by changing the file name
mypath ='C:\\Users\\YYJ\\Desktop\\FIN580\\Homework1\\VolatilityForecasting\\src\\'+filename+'\\'

onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]

prs = Presentation()
blank_slide_layout = prs.slide_layouts[6]

for i in range(len(onlyfiles)):
    img_path = mypath+onlyfiles[i]
    slide = prs.slides.add_slide(blank_slide_layout)
    left = Inches(0.8)
    height = Inches(5.5)
    pic = slide.shapes.add_picture(img_path, left, top, height=height)

prs.save(filename+'.pptx')