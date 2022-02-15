import os
import datetime as dtm


class FilesysHelper:
    """
    !mkdir empty_dir

    !touch empty_file

    !touch not_empty_file

    !echo "xxxx" > not_empty_file

    !mkdir not_empty_dir

    !cp -r empty_dir/ not_empty_dir/
    !cp  not_empty_file not_empty_dir/

    FH = FilesysHelper()
    print(FH.is_empty_dir("empty_dir/"))
    print(FH.is_empty_dir("./"))
    print(FH.is_empty_dir("not"))
    print(FH.is_empty_file("empty_file"))
    print(FH.is_empty_file("empty_dir/"))
    """

    def get_file_update_time_utc(self, path):
        return dtm.datetime.strftime(dtm.datetime.utcfromtimestamp(os.path.getmtime(path)),"%Y%m%d%H%M%S")

    def is_exist(self,path):
        return os.path.exists(path)
        
    def is_file(self,path):
        if not self.is_exist(path):
            print(f"No such file! {path}")
            return False
        return os.path.isfile(path)
    
    def is_dir(self,path):
        return os.path.isdir(path)
    
    def is_empty_file(self,path):
        if not self.is_exist(path):
            print(f"No such file! {path}")
            return False
        if not self.is_file(path):
            print(f"Not file type! {path}")
            return False
        else:
            return os.path.getsize(path) == 0

    def is_empty_dir(self,path):
        if not self.is_exist(path):
            print(f"No such dir! {path}")
            return False
        if not self.is_dir(path):
            print(f"Not dir type! {path}")
            return False
        else:
            return len(os.listdir(path)) == 0
