import click
import os
from osfclient.api import OSF
import tarfile

# this script does the same as (from terminal)
# osf -r -p your_password -u your_username upload local_path remote_path

LOCAL_PATH = 'path/data'  # local path to the data
LOCAL_PATH = '/home/maja/Desktop/temp/ramp_challenge'
REMOTE_PATH = 'test'  # remote path where to store the data on OSF
# if the data in this path already exists it will be overwritten
PROJECT_CODE_PUBLIC = 't4uf8'  # to find your PROJECT_CODE navigate to your OSF
# project on the web. The link will be something of this type:
# https://osf.io/t4uf8/ , here t4uf8 is the PROJECT_CODE
PROJECT_CODE_PRIVATE = 'vw8sh'


@click.command()
@click.option(
    "--username", required=True,
    help="Your username to the private repository"
)
@click.option(
    "--password", required=True,
    help="Your password to the private repository"
)
def upload_recursive_to_osf(username, password):
    # here we are only using recursive
    if not os.path.isdir(LOCAL_PATH):
        raise RuntimeError(f"Expected source ({LOCAL_PATH})"
                           "to be a directory")
    osf = OSF(username=username, password=password)

    # ########################################################
    # TODO: make the split to public and private data directories
    # to have a path:
    # LOCAL_PATH
    #       |---public
    #       |---private
    # all the data in the public directory will be added to the
    # public repo, and from private directory to the private repo
    #
    # here the split has already been done beforehand

    project_codes = [PROJECT_CODE_PUBLIC, PROJECT_CODE_PRIVATE]
    project_types = ['public', 'private']

    for project_code, project_type in zip(project_codes, project_types):

        print(f'compressing {project_type} data')
        used_dir = os.path.join(LOCAL_PATH, project_type)
        tar_name = os.path.join(LOCAL_PATH, project_type + '.tar.gz')

        # add files from the given dir to your archive
        def _add_dir_to_archive(path, tar_name):
            with tarfile.open(tar_name, "w:gz") as tar_handle:
                for root, dirs, files in os.walk(path):
                    local_dir = os.path.relpath(root, path)
                    if local_dir == '.':
                        local_dir = ''
                    for file in files:
                        tar_handle.add(os.path.join(root, file),
                                       arcname=os.path.join(local_dir, file))
        _add_dir_to_archive(used_dir, tar_name)
        print(f'uploading {project_type} data')

        # establish the connection with the correct repo on osf
        project = osf.project(project_code)
        store = project.storage('osfstorage')

        with open(tar_name, 'rb') as fp:
            fname = os.path.join(REMOTE_PATH, project_type + '.tar.gz')
            store.create_file(fname, fp, force=True)
        print(f'successfully uploaded {fname} to {REMOTE_PATH}')


if __name__ == "__main__":
    upload_recursive_to_osf()
