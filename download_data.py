"""Fetcher for RAMP data stored in OSF
To adapt it for another challenge, change the CHALLENGE_NAME and upload
public/private data as `tar.gz` archives in dedicated OSF folders named after
the challenge.
"""
import tarfile
import argparse
from zlib import adler32
from pathlib import Path
from osfclient.api import OSF
from osfclient.exceptions import UnauthorizedException

LOCAL_DATA = Path(__file__).parent / "data"

CHALLENGE_NAME = 'test'
# you might choosing checking for the correct checksum, if not set
# data_checksum to None
RAMP_FOLDER_CONFIGURATION = {
    'public': dict(
        code='t4uf8', archive_name='public.tar.gz',
        data_checksum=None  #1762354359
    ),
    'private': dict(
        code='vw8sh', archive_name='private.tar.gz',
        data_checksum=None  # 2915900195
    ),
}
DOWNLOAD_URL = "https://plmbox.math.cnrs.fr/f/8224e749026747758c56/?dl=1"


def get_connection_info(get_private, username=None, password=None):
    "Get connection to OSF and info relative to public/private data."
    if get_private:
        osf, folder_name = OSF(username=username, password=password), 'private'
    else:
        assert username is None and password is None, (
            "Username and password should only be provided when fetching "
            "private data."
        )
        osf, folder_name = OSF(), 'public'
    data_config = RAMP_FOLDER_CONFIGURATION[folder_name]

    try:
        project = osf.project(data_config['code'])
        store = project.storage('osfstorage')
    except UnauthorizedException:
        raise ValueError("Invalid credentials for RAMP private storage.")
    return store, data_config


def get_one_element(container, name):
    "Get one element from OSF container with a comprehensible failure error."
    elements = [f for f in container if f.name == name]
    container_name = (
        container.name if hasattr(container, 'name') else CHALLENGE_NAME
    )
    assert len(elements) == 1, (
        f'There is no element named {name} in {container_name} from the RAMP '
        'OSF account.'
    )
    return elements[0]


def hash_folder(folder_path):
    """Return the Adler32 hash of an entire directory."""
    folder = Path(folder_path)

    # Recursively scan the folder and compute a checksum
    checksum = 1
    for f in sorted(folder.rglob('*')):
        if f.is_file():
            checksum = adler32(f.read_bytes(), checksum)
        else:
            checksum = adler32(f.name.encode(), checksum)

    return checksum


def checksum_data(private, raise_error=False):
    folder = 'private' if private else 'public'
    data_checksum = RAMP_FOLDER_CONFIGURATION[folder]['data_checksum']
    if data_checksum:
        local_checksum = hash_folder(LOCAL_DATA)
        if raise_error and data_checksum != local_checksum:
            raise ValueError(
                f"The checksum does not match. Expecting {data_checksum} but "
                f"got {local_checksum}. The archive seems corrupted. Try to "
                f"remove {LOCAL_DATA} and re-run this command."
            )

        return data_checksum == local_checksum
    else:
        True


def download_from_osf(private, username=None, password=None):
    "Download and uncompress the data from OSF."

    # check if data directory is empty
    if not LOCAL_DATA.exists() or not any(LOCAL_DATA.iterdir()):
        LOCAL_DATA.mkdir(exist_ok=True)

        print("Checking the data URL...", end='', flush=True)
        # Get the connection to OSF
        store, data_config = get_connection_info(
            private, username=username, password=password
        )

        # Find the folder in the OSF project
        challenge_folder = get_one_element(store.folders, CHALLENGE_NAME)

        # Find the file to download from the OSF project
        archive_name = data_config['archive_name']
        osf_file = get_one_element(challenge_folder.files, archive_name)
        print('Ok.')

        # Download the archive in the data
        ARCHIVE_PATH = LOCAL_DATA / archive_name
        print("Downloading the data...")
        with open(ARCHIVE_PATH, 'wb') as f:
            osf_file.write_to(f)

        # Uncompress the data in the data folder
        print("Extracting now...", end="", flush=True)
        with tarfile.open(ARCHIVE_PATH) as tf:
            tf.extractall(LOCAL_DATA)
        print("Ok.")

        # Clean the directory by removing the archive
        print("Removing the archive...", end="", flush=True)
        ARCHIVE_PATH.unlink()
        print("Ok.")
        print("Checking the data...", end='', flush=True)
        checksum_data(private, raise_error=True)
        print("Ok.")
    else:
        print(f'{LOCAL_DATA} directory is not empty. Please empty it or select'
              ' another destination for LOCAL_DATA if you wish to proceed')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=f'Data loader for the {CHALLENGE_NAME} challenge on RAMP.'
    )
    parser.add_argument('--private', action='store_true',
                        help='If this flag is used, download the private data '
                        'from OSF. This requires the username and password '
                        'options to be provided.')
    parser.add_argument('--username', type=str, default=None,
                        help='Username for downloading private OSF data.')
    parser.add_argument('--password', type=str, default=None,
                        help='Password for downloading private OSF data.')
    args = parser.parse_args()
    download_from_osf(args.private, args.username, args.password)
'''

import os
import click
from osfclient.api import OSF

# NOTE: we are not using the fetch_from_osf from ramp_utils.datasets because
# too many files are to be loaded (hence checking the id of each of them would
# be too time consuming)

# in the command line: osf -p t4uf8 clone temp/
# however this corresponds to the whole project. we are interested only in the
# MEG data here

# this script does the same as (from terminal)
# osf upload local_path remote_path

LOCAL_PATH = 'data'  # local path to the data
REMOTE_PATH = 'MEG/'  # remote path where to store the data on OSF
PROJECT_CODE = 't4uf8'  # to find your PROJECT_CODE navigate to your OSF
PROJECT_CODE_PRIVATE = 'vw8sh'
# project on the web. The link will be something of this type:
# https://osf.io/t4uf8/ , here t4uf8 is the PROJECT_CODE


@click.command()
@click.option("--private", is_flag=True)
@click.option(
    "--username",
    help="Your username to the private repository"
)
@click.option(
    "--password",
    help="Your password to the private repository"
)
def download_from_osf(private, username, password):
    file_idx = 0
    if private:
        project_code = PROJECT_CODE_PRIVATE
    else:
        project_code = PROJECT_CODE

    # if the file already exists it will overwrite it
    # osf = OSF(username=USERNAME, password=PASSWORD)
    if private:
        osf = OSF(username=username, password=password)
    else:
        osf = OSF()
    project = osf.project(project_code)
    store = project.storage('osfstorage')

    for file_ in store.files:
        # get only those files which are stored in REMOTE_PATH
        pathname = file_.path

        if REMOTE_PATH not in pathname:
            # we are not interested in this file
            continue
        # otherwise we are copying it locally
        # check if the directory tree exists and add the dirs if necessary

        # do not include project name
        pathname = pathname[pathname.find(REMOTE_PATH)+len(REMOTE_PATH):]
        save_file = os.path.join(LOCAL_PATH, pathname)
        pathfile, filename = os.path.split(save_file)

        if not os.path.exists(pathfile):
            os.makedirs(pathfile)

        if not os.path.exists(save_file):
            # do not save it if the file already exists
            with open(save_file, "wb") as f:
                file_.write_to(f)
            file_idx += 1
        else:
            print(f'Skipping existing file {save_file}')
    print(f'saved {file_idx} files to {LOCAL_PATH}')


if __name__ == "__main__":
    download_from_osf()
'''