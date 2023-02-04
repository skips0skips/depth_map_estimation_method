import glob, os


def rename(dir, pattern):
    i=1
    for pathAndFilename in glob.iglob(os.path.join(dir, pattern)):
        title, ext = os.path.splitext(os.path.basename(pathAndFilename))
        os.rename(pathAndFilename, 
                  os.path.join(dir, str(i) + ext))
        i+=1



        