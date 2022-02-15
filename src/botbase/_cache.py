import pickle
import os
import datetime as dtm
from . import logger, bash2py, mkdirs
from .config import BASE_DIR_cwd as BASE_DIR


class Cache:
    """
    data = {"a":1,"b":2}
    Cache.cache_data(data,"test_cache")
    data["b"] = 333
    Cache.cache_data(data,"test_cache",sub_dir='v2')
    data = Cache.reload_cache("test_cache")
    print(data["b"])
    data = Cache.reload_cache("test_cache",sub_dir='v2')
    print(data["b"])
    data = Cache.reload_cache("CACHE_test_cache.pkl",nm_marker=False)
    print(data["b"])
    # Cache.clear_cache(AreYouSure='clear_cache')
    """
    @staticmethod
    def cache_data(data,
                   nm_marker=None,
                   dt_format='%Y%m%d_%Hh',
                   sub_dir=None,
                   add_time=False):

        tm_str = dtm.datetime.now().strftime(dt_format)
        if nm_marker is not None:
            name_ = nm_marker
            if add_time:
                name_ += tm_str
        else:
            name_ = tm_str

        if sub_dir:
            if not isinstance(sub_dir, str):
                raise
            mkdirs(os.path.join(BASE_DIR, 'cached_data', sub_dir))
            path_ = os.path.join(BASE_DIR, 'cached_data', sub_dir,
                                 f'CACHE_{name_}.pkl')

        else:
            mkdirs(os.path.join(BASE_DIR, 'cached_data'))
            path_ = os.path.join(BASE_DIR, f'cached_data/CACHE_{name_}.pkl')
        with open(path_, 'wb') as file:
            pickle.dump(data, file)
        logger.info(f'Cache Successfully! File name: {path_}')

    @staticmethod
    def reload_cache(file_nm="",
                     nm_marker=True,
                     sub_dir=None,
                     prefix='CACHE_',
                     postfix='.pkl'):

        if nm_marker:  # not including pre/post fix, we should add it
            file_nm = prefix + file_nm + postfix
        if sub_dir is None:
            base_dir = os.path.join(BASE_DIR, 'cached_data')
        else:
            base_dir = os.path.join(BASE_DIR, 'cached_data', sub_dir)
        file_path = os.path.join(base_dir, file_nm)
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        logger.info(f'Successfully Reload: {file_path}')
        return data

    @staticmethod
    def clear_cache(AreYouSure):
        if AreYouSure == 'clear_cache':
            bash2py(f"cd {BASE_DIR} && rm -rf cached_data/")
        else:
            logger.info(
                "If you truely wanna clear all cached data, set AreYouSure as 'clear_cache'"
            )

    @staticmethod
    def list_cache():
        """
        from arsenal import Cache
        res = Cache.list_cache()
        print(res)
        """
        SHELL = f"cd {BASE_DIR} && ls cached_data/"
        res = bash2py(SHELL)
        return res

    @staticmethod
    def dump_pkl(data, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(data, file)
        logger.info(f'Cache Successfully! File name: {file_path}')

    @staticmethod
    def load_pkl(file_path):
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        logger.info(f'Successfully Reload: {file_path}')
        return data
