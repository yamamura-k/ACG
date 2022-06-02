from collections import defaultdict

import pickle
import os
import torch

from utils.common import defaultdict2dict
from utils.logging import setup_logger

logger = setup_logger(__name__)


class Singleton(object):
    """
    
    """

    def __new__(cls, *args, **kargs):
        if not hasattr(cls, "_instance"):
            cls._instance = super(Singleton, cls).__new__(cls)
        return cls._instance


class GatherManager(Singleton):
    """Gather

    Attributes
    ----------
    listofGather: dict
        Gathergather
        
    """

    listofGather = dict()

    def getGather(self, gathername):
        """Gather

        Parameters
        ----------
        gathername: str
             gather

        Return
        ------
        Gather
            return Gather Class

        """
        if gathername in self.listofGather:
            return self.listofGather[gathername]
        newGather = Gather(gathername)
        self.listofGather[gathername] = newGather
        return newGather


class Gather:
    """.

    Parameters
    ----------
    name: str
        GatherGather

    Attributes
    ----------
    info: dict or default dict
        key"name"setParamkey
        valuename__init__name,
        setParamvalue.

    Notes
    -----
    """

    def __init__(self, name):
        self.info = defaultdict()
        self.name = name
        self.info["name"] = name

    def clearDict(self):
        """dict"""
        self.info.clear()

    def getDict(self):
        """gatherDict

        Returns
        -------
        dict
            Dict

        """
        return self.info

    def setDict(self, dicts):
        """dict（）

        Parameters
        ----------
        dicts: dict
            dict

        """
        self.info = dicts

    def save2pickle(self, savepath):
        """dictpickle

        Parameters
        ----------
        savepath: str

        """
        dirname = os.path.dirname(savepath)
        assert not os.path.exists(savepath), "Statics File already Exist!"
        os.makedirs(dirname, exist_ok=True)
        logger.info(f"\n [SAVE]: {savepath}")
        with open(savepath, "wb") as picklefile:
            pickle.dump(self.getDict(), picklefile, 2)

    def save2pth(self, savepath):
        """dictpth

        Parameters
        ----------
        savepath: str

        """
        assert not os.path.exists(savepath), "Statics File already Exist!"
        dirname = os.path.dirname(savepath)
        logger.info(f"\n [SAVE]: {savepath}")

        os.makedirs(dirname, exist_ok=True)
        torch.save(self.getDict(), savepath)

    def load(self, filename):
        """(pickle)dict

        Parameters
        ----------
        filename: str
            
        """
        with open(filename, "rb") as picklefile:
            load_dict = pickle.load(picklefile)
            self.info = load_dict
            self.name = load_dict["name"]
            print("load finish")

    def setParam(self, key, value):
        """dict

        Parameters
        ----------
        key: str
            dictkey

        value: object
            dict

        Notes
        -----
        key
        dictupdateDict()
        """
        # assert
        assert not key in self.info.keys(), "Key already exist!"
        self.info[key] = value

    def updateDict(self, key, dd):
        """dict

        Parameters
        ----------
        key: str
            dictkey

        dd: dict
            
        """
        assert type(self.info[key]) is not dict(), f"{key} is not a dict"
        assert type(dd) is not dict(), f"{str(dd)} is not a dict"
        self.info[key].update(dd)

    def setForceSave2pickle(self, savepath):
        """dictpickle

        Parameters
        ----------
        savepath: str

        """
        dirname = os.path.dirname(savepath)
        os.makedirs(dirname, exist_ok=True)
        info = defaultdict2dict(self.info)
        logger.info(f"\n [SAVE]: {savepath}")

        with open(savepath, "wb") as picklefile:
            pickle.dump(info, picklefile, 2)

    def setForceSave2pth(self, savepath):
        """dictpth

        Parameters
        ----------
        savepath: str

        """
        dirname = os.path.dirname(savepath)
        logger.info(f"\n [SAVE]: {savepath}")

        os.makedirs(dirname, exist_ok=True)
        torch.save(self.getDict(), savepath)

    def checksavefile(self, filepath):
        """

        Parameters
        ----------
        filepath: str

        """
        assert not os.path.exists(filepath), "Statics File already Exist!"


def testGather():
    gather1 = GatherManager().getGather("gather1")
    gather2 = GatherManager().getGather("gather1")
    gather1.setParam("key", 1)
    gather2.setParam("key2", 2)

    print(gather1.getDict())
    print(gather2.getDict())
    print(gather1)
    print(gather2)
