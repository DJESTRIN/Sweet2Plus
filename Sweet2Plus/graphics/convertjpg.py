from PIL import Image
import glob

def convert(search_root):
    files = glob.glob(search_root)
    for file in files:
        jpg_image = Image.open(file)
        file,_=file.split('.j')
        file=file+'.png'
        jpg_image.save(file)

if __name__=='__main__':
    convert(r"C:\Users\listo\Sweet2Plus\images\*.jpg")