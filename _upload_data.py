import click
import os
from osfclient.api import OSF

# this script does the same as (from terminal)
# osf -r -p your_password -u your_username upload local_path remote_path

LOCAL_PATH = 'meg/data'  # local path to the data
REMOTE_PATH = 'meg'  # remote path where to store the data on OSF
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
    # all the data in the public directory will be recursively added to the
    # public repo, and from private directory to the private repo
    # if the file already exists in the repo it will be overwritten

    # here the split has already been done after the data simulation

    project_codes = [PROJECT_CODE_PUBLIC, PROJECT_CODE_PRIVATE]
    project_types = ['public', 'private']

    for project_code, project_type in zip(project_codes, project_types):
        print(f'uploading {project_type} data')
        local_path_type = os.path.join(LOCAL_PATH, project_type)
        # establish the connection with the correct repo os osf
        project = osf.project(project_code)
        store = project.storage('osfstorage')

        _, dir_name = os.path.split(local_path_type)

        idx = 1
        for root, _, files in os.walk(local_path_type):
            subdir_path = os.path.relpath(root, local_path_type)
            for fname in files:
                local_path = os.path.join(root, fname)

                print(f'{idx} uploading: {local_path}')
                idx += 1
                with open(local_path, 'rb') as fp:
                    name = os.path.join(REMOTE_PATH, subdir_path, fname)
                    store.create_file(name, fp, force=True)

        print(f'uploaded {idx-1} files to {subdir_path}')


if __name__ == "__main__":
    upload_recursive_to_osf()
