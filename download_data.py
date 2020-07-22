import os

from ramp_utils.datasets import fetch_from_osf
from ramp_utils.datasets import OSFRemoteMetaData


PATH_DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data"
)
OSF_ARCHIVE = [
    OSFRemoteMetaData(
        filename="subject_10_lead_field.npz",
        id="j9b6n",
        revision=1,
    ),
    OSFRemoteMetaData(
        filename="subject_1_lead_field.npz",
        id="fgdzp",
        revision=1,
    ),
    OSFRemoteMetaData(
        filename="subject_2_lead_field.npz",
        id="r83sa",
        revision=1,
    ),
    OSFRemoteMetaData(
        filename="subject_3_lead_field.npz",
        id="vcp4g",
        revision=1,
    ),
    OSFRemoteMetaData(
        filename="subject_4_lead_field.npz",
        id="snwka",
        revision=1,
    ),
    OSFRemoteMetaData(
        filename="subject_5_lead_field.npz",
        id="uh6gm",
        revision=1,
    ),

    OSFRemoteMetaData(
        filename="subject_6_lead_field.npz",
        id="mz3bk",
        revision=1,
    ),
    OSFRemoteMetaData(
        filename="subject_7_lead_field.npz",
        id="jhgx4",
        revision=1,
    ),
    OSFRemoteMetaData(
        filename="subject_8_lead_field.npz",
        id="fa3me",
        revision=1,
    ),
    OSFRemoteMetaData(
        filename="subject_9_lead_field.npz",
        id="5sy72",
        revision=1,
    )
]

TEST_OSF_ARCHIVE = [
    OSFRemoteMetaData(
        filename="target.npz",
        id="jc2r3",
        revision=1,
    ),
    OSFRemoteMetaData(
        filename="X.csv.gz",
        id="7ywn3",
        revision=1,
    )
]

TRAIN_OSF_ARCHIVE = [
    OSFRemoteMetaData(
        filename="target.npz",
        id="9vrb6",
        revision=1,
    ),
    OSFRemoteMetaData(
        filename="X.csv.gz",
        id="4ufyj",
        revision=1,
    )
]


def download_data():
    fetch_from_osf(path_data=PATH_DATA, metadata=OSF_ARCHIVE)

    fetch_from_osf(path_data=os.path.join(PATH_DATA, 'test'),
                   metadata=TEST_OSF_ARCHIVE)

    fetch_from_osf(path_data=os.path.join(PATH_DATA, 'train'),
                   metadata=TRAIN_OSF_ARCHIVE)


if __name__ == "__main__":
    download_data()
