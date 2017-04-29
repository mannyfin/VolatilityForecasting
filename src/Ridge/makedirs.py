import os


def makedirs(root_folder_path=str, methodkind=str, name=str):
    if os.path.isdir(root_folder_path+'//'+methodkind+'//'+name) is False:
        os.chdir(root_folder_path)
        if os.path.isdir(methodkind) is False:
            os.mkdir(methodkind)
        os.chdir(methodkind)
        if os.path.isdir(methodkind) is False:
            os.mkdir(name)
        os.chdir(name)
        # os.chdir('../../..')
    else:
        # change to the dir
        os.chdir(root_folder_path+'//'+methodkind+'//'+name)
