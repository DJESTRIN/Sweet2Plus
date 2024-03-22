import tifffile
import glob,os
def gen_tiff_stack(input_dir=r'C:\Users\listo\tmtassay\TMTAssay\Day1\twophoton\24-3-18\24-3-18_C4620081_M1_R1-058'):
    with tifffile.TiffWriter(r'C:\Users\listo\tmtassay\TMTAssay\Day1\twophoton\24-3-18\24-3-18_C4620081_M1_R1-058.tif') as stack:
        for filename in glob.glob(os.path.join(input_dir,'*.tif')):
            stack.save(
                tifffile.imread(filename), 
                photometric='minisblack', 
                contiguous=True
            )

if __name__=='__main__':
    gen_tiff_stack()