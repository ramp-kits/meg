import os

from ramp_utils.datasets import fetch_from_osf
from ramp_utils.datasets import OSFRemoteMetaData


PATH_DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "data"
)
OSF_ARCHIVE = [
    OSFRemoteMetaData(
        filename="CC120166_lead_field.npz",
        id="tkb3n",
        revision=1,
    ),
    OSFRemoteMetaData(
        filename="CC120182_lead_field.npz",
        id="vmws3",
        revision=1,
    ),
    OSFRemoteMetaData(
        filename="CC120218_lead_field.npz",
        id="kvp48",
        revision=1,
    ),
    OSFRemoteMetaData(
        filename="CC120264_lead_field.npz",
        id="3ej8h",
        revision=1,
    ),
    OSFRemoteMetaData(
        filename="CC120309_lead_field.npz",
        id="qpg48",
        revision=1,
    ),
    OSFRemoteMetaData(
        filename="CC120313_lead_field.npz",
        id="jt3wy",
        revision=1,
    ),

    OSFRemoteMetaData(
        filename="CC120319_lead_field.npz",
        id="esjtd",
        revision=1,
    ),
    OSFRemoteMetaData(
        filename="CC120376_lead_field.npz",
        id="n24zt",
        revision=1,
    ),
    OSFRemoteMetaData(
        filename="CC120469_lead_field.npz",
        id="zdxhk",
        revision=1,
    ),
    OSFRemoteMetaData(
        filename="CC120550_lead_field.npz",
        id="q3w9h",
        revision=1,
    )
]

TEST_OSF_ARCHIVE = [
    OSFRemoteMetaData(
        filename="target.npz",
        id="mj46z",
        revision=1,
    ),
    OSFRemoteMetaData(
        filename="X.csv",
        id="7sx6r",
        revision=1,
    )
]

TRAIN_OSF_ARCHIVE = [
    OSFRemoteMetaData(
        filename="target.npz",
        id="5zyvc",
        revision=1,
    ),
    OSFRemoteMetaData(
        filename="X.csv",
        id="s7y25",
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
